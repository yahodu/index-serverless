"""
Face Search Server — MongoDB backed
=====================================
POST /index   — extract embeddings from an image URL, store in MongoDB
POST /search  — find all indexed images containing the query face from URL
GET  /health  — server status

All operations are timed and logged in detail.
"""

import os
import time
import logging
import traceback

import cv2
import numpy as np
import uvicorn
import httpx
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
from insightface.app import FaceAnalysis

# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("face-server")

# ─── Config from env ──────────────────────────────────────────────────────────

MODEL_NAME   = os.environ.get("MODEL_NAME", "buffalo_l")
MONGO_URI    = os.environ.get("MONGO_URI")            # mongodb+srv://...
MONGO_DB     = os.environ.get("MONGO_DB", "facedb")
MONGO_COLL   = os.environ.get("MONGO_COLL", "embeddings")
PORT         = int(os.environ.get("PORT", 18000))
DOWNLOAD_TIMEOUT = int(os.environ.get("DOWNLOAD_TIMEOUT", 30))  # seconds

# ─── App & globals ────────────────────────────────────────────────────────────

app      = FastAPI()
face_app = None
mongo_collection = None
http_client = None


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    global face_app, mongo_collection, http_client

    # ── HTTP Client ───────────────────────────────────────────────────────────
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(DOWNLOAD_TIMEOUT),
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    log.info(f"HTTP client initialized | timeout={DOWNLOAD_TIMEOUT}s")

    # ── Load InsightFace model ────────────────────────────────────────────────
    log.info(f"Loading InsightFace model: {MODEL_NAME}")
    t0 = time.perf_counter()
    face_app = FaceAnalysis(
        name=MODEL_NAME,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    model_load_ms = (time.perf_counter() - t0) * 1000
    log.info(f"Model loaded | model={MODEL_NAME} | time={model_load_ms:.1f}ms")

    # ── Connect to MongoDB ────────────────────────────────────────────────────
    if not MONGO_URI:
        log.error("MONGO_URI environment variable not set")
        raise RuntimeError("MONGO_URI not set")

    log.info("Connecting to MongoDB...")
    t0 = time.perf_counter()
    client = MongoClient(MONGO_URI)
    mongo_collection = client[MONGO_DB][MONGO_COLL]
    # Index on image_path for fast lookups
    mongo_collection.create_index([("image_path", ASCENDING)])
    mongo_ms = (time.perf_counter() - t0) * 1000
    log.info(f"MongoDB connected | db={MONGO_DB} | collection={MONGO_COLL} | time={mongo_ms:.1f}ms")

    # ── Done ──────────────────────────────────────────────────────────────────
    total_docs = mongo_collection.count_documents({})
    log.info(f"Startup complete | existing_docs={total_docs}")
    print("Application startup complete.")   # Vast.ai watches for this exact line


@app.on_event("shutdown")
async def shutdown():
    if http_client:
        await http_client.aclose()
        log.info("HTTP client closed")


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def download_image(url: str) -> tuple[bytes | None, str | None]:
    """
    Download image from URL.
    Returns: (image_bytes, error_message)
    """
    try:
        response = await http_client.get(url)
        response.raise_for_status()
        return response.content, None
    except httpx.TimeoutException:
        return None, f"Download timeout after {DOWNLOAD_TIMEOUT}s"
    except httpx.HTTPStatusError as e:
        return None, f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
    except httpx.RequestError as e:
        return None, f"Request failed: {str(e)}"
    except Exception as e:
        return None, f"Download error: {str(e)}"


def decode_image(contents: bytes) -> np.ndarray | None:
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def timing(label: str, t0: float) -> float:
    """Returns elapsed ms and logs it."""
    ms = (time.perf_counter() - t0) * 1000
    log.info(f"  {label:<30} {ms:>8.2f} ms")
    return ms


# ─── /index ───────────────────────────────────────────────────────────────────

@app.post("/index")
async def index_image(
    image_url: str  = Form(...),            # Storj object URL
    image_path: str = Form(None),           # logical path/name (defaults to URL if not provided)
    overwrite: bool = Form(False),          # if True, replace existing doc for this path
):
    # Use URL as image_path if not provided
    if image_path is None:
        image_path = image_url
    
    request_id = f"idx-{int(time.time()*1000)}"
    log.info(f"[{request_id}] /index | image_url={image_url} | image_path={image_path} | overwrite={overwrite}")
    t_request = time.perf_counter()

    # ── Download image ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    contents, error = await download_image(image_url)
    download_ms = timing("image download", t0)

    if error:
        log.warning(f"[{request_id}] Download failed: {error}")
        return JSONResponse({"error": error}, status_code=400)

    log.info(f"[{request_id}] Downloaded {len(contents)} bytes")

    # ── Decode ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    img = decode_image(contents)
    decode_ms = timing("image decode", t0)

    if img is None:
        log.warning(f"[{request_id}] Could not decode image")
        return JSONResponse({"error": "Cannot decode image"}, status_code=400)

    h, w = img.shape[:2]
    log.info(f"[{request_id}] Image size: {w}x{h}")

    # ── Face detection + embedding ────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        faces = face_app.get(img)
    except Exception as e:
        log.error(f"[{request_id}] Inference failed: {e}")
        return JSONResponse({"error": f"Inference error: {e}"}, status_code=500)
    inference_ms = timing("detection + embedding", t0)

    if not faces:
        log.info(f"[{request_id}] No faces detected")
        return JSONResponse({
            "request_id": request_id,
            "image_path": image_path,
            "image_url": image_url,
            "faces_detected": 0,
            "indexed": 0,
            "timings_ms": {
                "download": round(download_ms, 2),
                "decode": round(decode_ms, 2),
                "inference": round(inference_ms, 2)
            },
        })

    log.info(f"[{request_id}] Faces detected: {len(faces)}")

    # ── Save to MongoDB ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    docs_written = 0
    try:
        if overwrite:
            mongo_collection.delete_many({"image_path": image_path})

        docs = []
        for i, face in enumerate(faces):
            if face.normed_embedding is None:
                continue
            docs.append({
                "image_path": image_path,
                "image_url": image_url,
                "face_index": i,
                "embedding": face.normed_embedding.tolist(),   # list of 512 floats
                "det_score": float(face.det_score),
                "bbox": face.bbox.tolist(),
                "model": MODEL_NAME,
                "image_size": [w, h],
            })

        if docs:
            mongo_collection.insert_many(docs)
            docs_written = len(docs)

    except PyMongoError as e:
        log.error(f"[{request_id}] MongoDB write failed: {e}")
        return JSONResponse({"error": f"MongoDB error: {e}"}, status_code=500)

    mongo_ms = timing("mongodb write", t0)
    total_ms = (time.perf_counter() - t_request) * 1000

    log.info(
        f"[{request_id}] /index done | "
        f"faces={len(faces)} | written={docs_written} | "
        f"total={total_ms:.2f}ms "
        f"[download={download_ms:.1f} decode={decode_ms:.1f} inference={inference_ms:.1f} mongo={mongo_ms:.1f}]"
    )

    return {
        "request_id": request_id,
        "image_path": image_path,
        "image_url": image_url,
        "faces_detected": len(faces),
        "indexed": docs_written,
        "timings_ms": {
            "download": round(download_ms, 2),
            "decode": round(decode_ms, 2),
            "inference": round(inference_ms, 2),
            "mongo_write": round(mongo_ms, 2),
            "total": round(total_ms, 2),
        },
    }


# ─── /search ──────────────────────────────────────────────────────────────────

@app.post("/search")
async def search(
    image_url: str   = Form(...),          # Query image URL
    threshold: float = Form(0.45),
):
    request_id = f"srch-{int(time.time()*1000)}"
    log.info(f"[{request_id}] /search | image_url={image_url} | threshold={threshold}")
    t_request = time.perf_counter()

    # ── Download image ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    contents, error = await download_image(image_url)
    download_ms = timing("image download", t0)

    if error:
        log.warning(f"[{request_id}] Download failed: {error}")
        return JSONResponse({"error": error}, status_code=400)

    log.info(f"[{request_id}] Downloaded {len(contents)} bytes")

    # ── Decode ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    img = decode_image(contents)
    decode_ms = timing("image decode", t0)

    if img is None:
        return JSONResponse({"error": "Cannot decode image"}, status_code=400)

    # ── Query embedding ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        faces = face_app.get(img)
    except Exception as e:
        log.error(f"[{request_id}] Inference failed: {e}")
        return JSONResponse({"error": f"Inference error: {e}"}, status_code=500)
    inference_ms = timing("query inference", t0)

    if not faces:
        log.info(f"[{request_id}] No face in query image")
        return JSONResponse({"error": "No face detected in query image"}, status_code=400)

    if len(faces) > 1:
        log.info(f"[{request_id}] Multiple faces in query ({len(faces)}), using highest confidence")
    query_face = sorted(faces, key=lambda f: f.det_score, reverse=True)[0]
    query_emb  = np.array(query_face.normed_embedding)
    log.info(f"[{request_id}] Query det_score={query_face.det_score:.4f}")

    # ── Load embeddings from MongoDB ──────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        cursor = mongo_collection.find(
            {},
            {"image_path": 1, "image_url": 1, "embedding": 1, "det_score": 1, "face_index": 1}
        )
        all_docs = list(cursor)
    except PyMongoError as e:
        log.error(f"[{request_id}] MongoDB read failed: {e}")
        return JSONResponse({"error": f"MongoDB error: {e}"}, status_code=500)
    mongo_read_ms = timing("mongodb load", t0)

    if not all_docs:
        log.info(f"[{request_id}] Index is empty")
        return {"matches": [], "total_searched": 0}

    log.info(f"[{request_id}] Loaded {len(all_docs)} embeddings from MongoDB")

    # ── Matmul similarity search ───────────────────────────────────────────────
    t0 = time.perf_counter()
    matrix = np.array([doc["embedding"] for doc in all_docs], dtype=np.float32)  # (N, 512)
    sims   = matrix @ query_emb.astype(np.float32)                                # (N,)
    matmul_ms = timing("matmul search", t0)

    # ── Filter + deduplicate ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    seen = {}
    for i, sim in enumerate(sims):
        if sim >= threshold:
            doc  = all_docs[i]
            path = doc["image_path"]
            if path not in seen or sim > seen[path]["similarity"]:
                seen[path] = {
                    "image_path": path,
                    "image_url": doc.get("image_url", path),
                    "similarity": round(float(sim), 4),
                    "det_score":  round(doc["det_score"], 4),
                    "face_index": doc["face_index"],
                }
    matches = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)
    filter_ms = timing("filter + dedup", t0)

    total_ms = (time.perf_counter() - t_request) * 1000

    log.info(
        f"[{request_id}] /search done | "
        f"searched={len(all_docs)} | matches={len(matches)} | "
        f"total={total_ms:.2f}ms "
        f"[download={download_ms:.1f} decode={decode_ms:.1f} inference={inference_ms:.1f} "
        f"mongo_load={mongo_read_ms:.1f} matmul={matmul_ms:.2f} filter={filter_ms:.2f}]"
    )

    return {
        "request_id": request_id,
        "matches": matches,
        "total_searched": len(all_docs),
        "timings_ms": {
            "download":   round(download_ms, 2),
            "decode":     round(decode_ms, 2),
            "inference":  round(inference_ms, 2),
            "mongo_load": round(mongo_read_ms, 2),
            "matmul":     round(matmul_ms, 2),
            "filter":     round(filter_ms, 2),
            "total":      round(total_ms, 2),
        },
    }


# ─── /health ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        total = mongo_collection.count_documents({})
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "indexed_faces": total,
        }
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
