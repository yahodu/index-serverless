# probe_rec.py  — run this once to inspect the raw record layout
import struct
import numpy as np
import cv2

MAGIC = 0xced7230a
REC  = "datasets/ms1m_retinaface/train.rec"
IDX  = "datasets/ms1m_retinaface/train.idx"

# ── read index ────────────────────────────────────────────────
offsets = {}
with open(IDX, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 2:
            offsets[int(parts[0])] = int(float(parts[1]))

keys = sorted(offsets.keys())
print(f"Total keys: {len(keys)}")

# ── inspect first 3 records raw ───────────────────────────────
with open(REC, 'rb') as f:
    for ki, key in enumerate(keys[:3]):
        offset = offsets[key]
        f.seek(offset)

        magic_raw  = f.read(4)
        magic      = struct.unpack_from('<I', magic_raw)[0]
        enc_raw    = f.read(4)
        encoded    = struct.unpack_from('<I', enc_raw)[0]
        data_len   = (encoded & 0x1FFFFFFF) * 4
        cflag      = encoded >> 29

        # read the remaining header bytes (16 bytes: index + id)
        hdr_rest   = f.read(16)
        index_val  = struct.unpack_from('<Q', hdr_rest, 0)[0]
        id_val     = struct.unpack_from('<Q', hdr_rest, 8)[0]

        payload_len = data_len - 16
        payload     = f.read(payload_len)

        print(f"\n── Record key={key} offset={offset} ──")
        print(f"  magic      : 0x{magic:08x}  {'OK' if magic == MAGIC else 'BAD'}")
        print(f"  cflag      : {cflag}")
        print(f"  data_len   : {data_len}")
        print(f"  payload_len: {payload_len}")
        print(f"  index_val  : {index_val}")
        print(f"  id_val     : {id_val}")

        # Show first 64 bytes of payload as hex so we can find the JPEG SOI marker (FF D8)
        print(f"  payload hex (first 64 bytes):")
        for row in range(0, min(64, len(payload)), 16):
            chunk = payload[row:row+16]
            hex_str  = ' '.join(f'{b:02x}' for b in chunk)
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
            print(f"    {row:3d}: {hex_str:<47}  {ascii_str}")

        # Search for JPEG SOI marker (FF D8 FF) anywhere in payload
        soi = payload.find(b'\xff\xd8\xff')
        print(f"  JPEG SOI (ff d8 ff) found at payload byte: {soi}")

        # Try to decode from the found offset
        if soi >= 0:
            buf = np.frombuffer(payload[soi:], dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                print(f"  imdecode SUCCESS at skip={soi}: shape={img.shape}")
            else:
                print(f"  imdecode FAILED even at SOI offset {soi}")







# The reader is finding the records (offsets look real, jpeg_len values are plausible) but `imdecode` is failing, which means we're slicing the payload at the wrong byte offset. The label header size varies by dataset. Let me add a probe to figure out the exact layout of your file:

# Run it:

# ```bash
# python probe_rec.py
# ```

# It will print something like:

# ```
# ── Record key=1 offset=0 ──
#   payload hex (first 64 bytes):
#     000: 00 00 00 00 00 00 00 00 00 00 80 3f ff d8 ff e0  ...........?....
#   JPEG SOI (ff d8 ff) found at payload byte: 12
#   imdecode SUCCESS at skip=12: shape=(112, 112, 3)
# ```

# The key number you're looking for is the **`JPEG SOI found at payload byte`** value. That is the exact skip offset to hardcode. Then paste that value here and I'll apply the one-line fix to `_read_record_at`:

# ```python
# # The only line that changes — replace 4 with whatever probe_rec.py reported:
# jpeg_bytes = payload[<SOI_OFFSET>:]
# ```

# Common values seen in the wild for InsightFace datasets are `4`, `8`, `12`, and `24` — which is why a hardcoded skip was unreliable. The probe script

