#!/usr/bin/env python3

import numpy as np
import zipfile
import os
import tempfile

# Load NLCD data
nlcd_data = np.load('data/data_32.npz')['train_data'][:, 0, :, :].flatten().astype(np.uint8)
nlcd_bytes = nlcd_data.tobytes()

# Load wiki data
with zipfile.ZipFile('data/wiki.train.raw.zip', 'r') as zf:
    wiki_bytes = zf.read(zf.namelist()[0])

# Compress NLCD
with tempfile.NamedTemporaryFile(suffix='.zip') as tmp:
    with zipfile.ZipFile(tmp.name, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr('data', nlcd_bytes)
    nlcd_ratio = len(nlcd_bytes) / os.path.getsize(tmp.name)

# Compress wiki
with tempfile.NamedTemporaryFile(suffix='.zip') as tmp:
    with zipfile.ZipFile(tmp.name, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr('data', wiki_bytes)
    wiki_ratio = len(wiki_bytes) / os.path.getsize(tmp.name)

print(f"NLCD compression ratio: {nlcd_ratio:.2f}:1")
print(f"Wiki compression ratio: {wiki_ratio:.2f}:1") 