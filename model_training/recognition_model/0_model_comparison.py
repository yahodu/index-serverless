"""
Face Search: Find all photos containing a specific face
========================================================
Compares antelopev2 vs buffalo_l on GPU.

Workflow:
  1. Index all faces in a directory (extract embeddings)
  2. Given a query photo (one face), find all matching photos

Usage:
    # Index + search in one go
    python face_search.py --dir ./photos --query query.jpg

    # Save index, reuse later
    python face_search.py --dir ./photos --query query.jpg --save-index index.npz

    # Load saved index (skip re-indexing)
    python face_search.py --load-index index.npz --query query.jpg

    # Tune threshold
    python face_search.py --dir ./photos --query query.jpg --threshold 0.45
"""

import argparse
import os
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("InsightFace not installed.")
    sys.exit(1)

# ─── Config ───────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
MODELS = ["buffalo_l", "antelopev2"]
DEFAULT_THRESHOLD = 0.45   # cosine similarity — tune this per your use case

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class FaceRecord:
    image_path: str
    face_index: int          # which face in the image (if multiple)
    embedding: np.ndarray
    det_score: float
    bbox: list               # [x1, y1, x2, y2]


@dataclass
class SearchResult:
    image_path: str
    face_index: int
    similarity: float
    det_score: float
    bbox: list


@dataclass
class ModelStats:
    model: str
    model_load_time_ms: float
    index_time_s: float
    avg_inference_ms: float        # avg per-image inference time during indexing
    query_inference_ms: float      # time to extract query embedding
    search_time_ms: float          # matmul + threshold filter
    total_faces_indexed: int
    total_images: int
    skipped_images: int
    results: list[SearchResult] = field(default_factory=list)


# ─── Core ─────────────────────────────────────────────────────────────────────

def load_model(model_name: str) -> tuple[FaceAnalysis, float]:
    print(f"  Loading {model_name}...", end=" ", flush=True)
    t0 = time.perf_counter()
    app = FaceAnalysis(
        name=model_name,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"done ({elapsed_ms:.0f} ms)")
    return app, elapsed_ms


def get_image_paths(directory: str) -> list[str]:
    paths = []
    for p in Path(directory).rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            paths.append(str(p))
    return sorted(paths)


def index_directory(app: FaceAnalysis, image_paths: list[str], model_name: str) -> tuple[list[FaceRecord], float, float]:
    """Extract embeddings from all faces in all images.
    Returns: records, total_index_time_s, avg_inference_ms_per_image
    """
    records = []
    skipped = 0
    inference_times = []

    print(f"\n  Indexing {len(image_paths)} images with {model_name}...")
    t0 = time.perf_counter()

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"    [SKIP] Cannot read: {path}")
            skipped += 1
            continue

        try:
            t_inf = time.perf_counter()
            faces = app.get(img)
            inference_times.append((time.perf_counter() - t_inf) * 1000)
        except Exception as e:
            print(f"    [SKIP] Error on {path}: {e}")
            skipped += 1
            continue

        for j, face in enumerate(faces):
            if face.normed_embedding is None:
                continue
            records.append(FaceRecord(
                image_path=path,
                face_index=j,
                embedding=face.normed_embedding.copy(),
                det_score=float(face.det_score),
                bbox=face.bbox.tolist(),
            ))

        # Progress every 20 images
        if (i + 1) % 20 == 0 or (i + 1) == len(image_paths):
            print(f"    {i+1}/{len(image_paths)} images | {len(records)} faces found", end="\r")

    total_s = time.perf_counter() - t0
    avg_inf_ms = float(np.mean(inference_times)) if inference_times else 0.0
    print(f"\n  Done. {len(records)} faces in {len(image_paths) - skipped} images "
          f"({total_s:.2f}s total | {avg_inf_ms:.1f}ms avg/image)")

    return records, total_s, avg_inf_ms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Both embeddings are already L2-normed, so dot product = cosine sim."""
    return float(np.dot(a, b))


def search(
    query_embedding: np.ndarray,
    records: list[FaceRecord],
    threshold: float,
) -> tuple[list[SearchResult], float]:
    """Find all faces above threshold, sorted by similarity."""
    t0 = time.perf_counter()

    # Vectorized cosine similarity (fast for large indexes)
    embeddings_matrix = np.stack([r.embedding for r in records])  # (N, 512)
    similarities = embeddings_matrix @ query_embedding              # (N,)

    search_ms = (time.perf_counter() - t0) * 1000

    results = []
    for i, sim in enumerate(similarities):
        if sim >= threshold:
            r = records[i]
            results.append(SearchResult(
                image_path=r.image_path,
                face_index=r.face_index,
                similarity=float(sim),
                det_score=r.det_score,
                bbox=r.bbox,
            ))

    # Sort by similarity descending
    results.sort(key=lambda x: x.similarity, reverse=True)

    # Deduplicate: keep best match per image
    seen = {}
    deduped = []
    for res in results:
        if res.image_path not in seen:
            seen[res.image_path] = res
            deduped.append(res)

    return deduped, search_ms


def get_query_embedding(app: FaceAnalysis, query_path: str, model_name: str) -> tuple[np.ndarray, float]:
    img = cv2.imread(query_path)
    if img is None:
        print(f"ERROR: Cannot read query image: {query_path}")
        sys.exit(1)

    t0 = time.perf_counter()
    faces = app.get(img)
    query_ms = (time.perf_counter() - t0) * 1000

    if not faces:
        print(f"ERROR: No face detected in query image: {query_path}")
        sys.exit(1)

    if len(faces) > 1:
        print(f"  Warning: {len(faces)} faces found in query image. Using the highest-confidence one.")
        faces = sorted(faces, key=lambda f: f.det_score, reverse=True)

    print(f"  Query face detected (det_score={faces[0].det_score:.4f}, inference={query_ms:.1f}ms)")
    return faces[0].normed_embedding, query_ms


# ─── Index save/load ──────────────────────────────────────────────────────────

def save_index(records: list[FaceRecord], path: str, model_name: str):
    embeddings = np.stack([r.embedding for r in records])
    image_paths = np.array([r.image_path for r in records])
    face_indices = np.array([r.face_index for r in records])
    det_scores = np.array([r.det_score for r in records])
    bboxes = np.array([r.bbox for r in records])
    np.savez_compressed(
        path,
        embeddings=embeddings,
        image_paths=image_paths,
        face_indices=face_indices,
        det_scores=det_scores,
        bboxes=bboxes,
        model_name=np.array([model_name]),
    )
    print(f"  Index saved to {path}")


def load_index(path: str) -> tuple[list[FaceRecord], str]:
    data = np.load(path, allow_pickle=False)
    model_name = str(data["model_name"][0])
    records = []
    for i in range(len(data["embeddings"])):
        records.append(FaceRecord(
            image_path=str(data["image_paths"][i]),
            face_index=int(data["face_indices"][i]),
            embedding=data["embeddings"][i],
            det_score=float(data["det_scores"][i]),
            bbox=data["bboxes"][i].tolist(),
        ))
    print(f"  Loaded index: {len(records)} faces (model: {model_name})")
    return records, model_name


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_results(stats: ModelStats, threshold: float):
    sep = "─" * 65
    print(f"\n{sep}")
    print(f"  Model : {stats.model}")
    print(f"  ┌─ Timing Breakdown ───────────────────────────────────────")
    print(f"  │  Model load       : {stats.model_load_time_ms:>8.1f} ms")
    print(f"  │  Indexing (total) : {stats.index_time_s:>8.2f} s   ({stats.total_images} images, {stats.skipped_images} skipped)")
    print(f"  │  Inference avg    : {stats.avg_inference_ms:>8.1f} ms  per image")
    print(f"  │  Query inference  : {stats.query_inference_ms:>8.1f} ms")
    print(f"  │  Search (matmul)  : {stats.search_time_ms:>8.3f} ms  ({stats.total_faces_indexed} faces)")
    print(f"  └──────────────────────────────────────────────────────────")
    print(f"  Faces indexed : {stats.total_faces_indexed}")
    print(f"  Threshold     : {threshold}")
    print(f"  Matches       : {len(stats.results)} photos contain the query face")
    print(sep)

    if not stats.results:
        print("  No matches found. Try lowering --threshold.")
        return

    print(f"  {'#':<4} {'Similarity':>10}  {'DetScore':>9}  {'Path'}")
    print(f"  {'-'*4} {'-'*10}  {'-'*9}  {'-'*40}")
    for i, res in enumerate(stats.results, 1):
        bar = "█" * int(res.similarity * 20)
        print(f"  {i:<4} {res.similarity:>10.4f}  {res.det_score:>9.4f}  {res.image_path}")
        print(f"       [{bar:<20}]")


def print_comparison(stats_list: list[ModelStats], threshold: float):
    print(f"\n{'═'*75}")
    print("  COMPARISON SUMMARY")
    print(f"{'═'*75}")
    print(f"  {'Model':<16} {'Load(ms)':>9} {'Index(s)':>9} {'Inf/img(ms)':>12} {'Query(ms)':>10} {'Search(ms)':>11} {'Matches':>8}")
    print(f"  {'-'*15} {'-'*9} {'-'*9} {'-'*12} {'-'*10} {'-'*11} {'-'*8}")
    for s in stats_list:
        print(f"  {s.model:<16} {s.model_load_time_ms:>9.1f} {s.index_time_s:>9.2f} "
              f"{s.avg_inference_ms:>12.1f} {s.query_inference_ms:>10.1f} "
              f"{s.search_time_ms:>11.3f} {len(s.results):>8}")

    # Overlap
    if len(stats_list) == 2:
        paths_a = {r.image_path for r in stats_list[0].results}
        paths_b = {r.image_path for r in stats_list[1].results}
        common = paths_a & paths_b
        only_a = paths_a - paths_b
        only_b = paths_b - paths_a
        print(f"\n  Agreement (both models matched): {len(common)} photos")
        if only_a:
            print(f"  Only {stats_list[0].model} matched: {len(only_a)} photos")
            for p in sorted(only_a):
                print(f"    - {p}")
        if only_b:
            print(f"  Only {stats_list[1].model} matched: {len(only_b)} photos")
            for p in sorted(only_b):
                print(f"    - {p}")

    print(f"{'═'*75}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Face Search: Find a person across a photo directory")
    parser.add_argument("--dir", help="Directory of photos to index")
    parser.add_argument("--query", required=True, help="Query photo (one face)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine similarity threshold (default: {DEFAULT_THRESHOLD}). "
                             f"Higher = stricter. Try 0.3-0.6.")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS,
                        help="Which models to run")
    parser.add_argument("--save-index", metavar="FILE",
                        help="Save computed index to .npz file for reuse")
    parser.add_argument("--load-index", metavar="FILE",
                        help="Load a previously saved index (skips indexing)")
    args = parser.parse_args()

    if not args.dir and not args.load_index:
        print("ERROR: Provide --dir to index photos or --load-index to load a saved index.")
        sys.exit(1)

    print(f"\n{'═'*65}")
    print("  Face Search  |  antelopev2 vs buffalo_l  |  GPU")
    print(f"{'═'*65}\n")

    all_stats = []

    for model_name in args.models:
        print(f"\n{'─'*65}")
        print(f"  [{model_name}]")
        print(f"{'─'*65}")

        app, load_ms = load_model(model_name)

        # ── Index ──────────────────────────────────────────────────────────
        if args.load_index:
            records, saved_model = load_index(args.load_index)
            if saved_model != model_name:
                print(f"  Warning: Index was built with '{saved_model}', now using '{model_name}'. "
                      f"Embeddings may not match!")
            stats = ModelStats(
                model=model_name,
                model_load_time_ms=load_ms,
                index_time_s=0,
                avg_inference_ms=0,
                query_inference_ms=0,
                search_time_ms=0,
                total_faces_indexed=len(records),
                total_images=0,
                skipped_images=0,
            )
        else:
            image_paths = get_image_paths(args.dir)
            if not image_paths:
                print(f"  No images found in {args.dir}")
                sys.exit(1)
            records, index_s, avg_inf_ms = index_directory(app, image_paths, model_name)

            if args.save_index:
                save_path = args.save_index.replace(".npz", f"_{model_name}.npz")
                save_index(records, save_path, model_name)

            stats = ModelStats(
                model=model_name,
                model_load_time_ms=load_ms,
                index_time_s=index_s,
                avg_inference_ms=avg_inf_ms,
                query_inference_ms=0,
                search_time_ms=0,
                total_faces_indexed=len(records),
                total_images=len(image_paths),
                skipped_images=len(image_paths) - len(records),  # rough
            )

        if not records:
            print("  No faces indexed. Check your image directory.")
            continue

        # ── Query ──────────────────────────────────────────────────────────
        print(f"\n  Processing query: {args.query}")
        query_emb, query_ms = get_query_embedding(app, args.query, model_name)
        stats.query_inference_ms = query_ms

        # ── Search ─────────────────────────────────────────────────────────
        print(f"  Searching {len(records)} face embeddings...")
        results, search_ms = search(query_emb, records, args.threshold)
        stats.search_time_ms = search_ms
        stats.results = results

        print_results(stats, args.threshold)
        all_stats.append(stats)

    if len(all_stats) > 1:
        print_comparison(all_stats, args.threshold)


if __name__ == "__main__":
    main()
