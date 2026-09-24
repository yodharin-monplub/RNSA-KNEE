"""Run ON THE RENTAL: stream one fastMRI single-coil knee archive and keep only compact uint8 image stacks.

Usage: build_fastmri_cache.py <archive>     (knee_singlecoil_val | knee_singlecoil_train)
       build_fastmri_cache.py merge         (after both: write manifest.json, copy the annotation CSVs)
Input : data/external/fastmri/links_knee.json (two signed NYU links; never logged), data/external/fastmri_plus/*.csv
Output: artifacts/model62/<archive>_XXX.npz (one array per annotated exam: slices x 320 x 320 uint8) and
        <archive>_manifest.json. The archive is decompressed as a stream through a reader that reconnects with an
        HTTP Range request when S3 drops the connection (the first attempt died on exactly that), so the raw
        k-space (94 GB) never touches the disk. Only exams listed by fastMRI+ are kept.
The NYU agreement forbids redistribution: the result goes to a PRIVATE Kaggle dataset for this project only.
"""
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

import h5py
import numpy as np
import requests

ROOT = Path('/workspace/rsna')
OUT = ROOT / 'artifacts/model62'
OUT.mkdir(parents=True, exist_ok=True)


class ResumingStream:
    """File-like sequential reader over an HTTP URL that resumes at the current offset after any failure."""

    def __init__(self, url):
        self.url, self.offset, self.response, self.reconnects = url, 0, None, 0

    def _open(self):
        self.response = requests.get(self.url, headers={'Range': f'bytes={self.offset}-'}, stream=True, timeout=60)
        assert self.response.status_code == 206, self.response.status_code
        self.raw = self.response.raw

    def read(self, size=-1):
        chunks, remaining = [], size
        while remaining:
            try:
                if self.response is None:
                    self._open()
                block = self.raw.read(min(remaining, 1 << 22))
                if not block:
                    total = int(self.response.headers['Content-Range'].rsplit('/', 1)[1])
                    if self.offset >= total:
                        break
                    raise IOError('short read')
                self.offset += len(block)
                remaining -= len(block)
                chunks.append(block)
            except Exception as exc:  # noqa: BLE001 - every network failure means: reconnect at the offset
                self.reconnects += 1
                print('reconnect', self.reconnects, 'at', self.offset, type(exc).__name__, flush=True)
                assert self.reconnects < 500
                self.response = None
                time.sleep(min(60, 2 * self.reconnects))
        return b''.join(chunks)


def build(archive):
    links = json.loads((ROOT / 'data/external/fastmri/links_knee.json').read_text())
    wanted = {line.strip().replace('.h5', '') for line in
              (ROOT / 'data/external/fastmri_plus/knee_file_list.csv').read_text().splitlines() if line.strip()}
    manifest, pending, shard_index, seen, started = {}, {}, 0, 0, time.time()

    def flush():
        nonlocal pending, shard_index
        if pending:
            name = f'{archive}_{shard_index:03d}.npz'
            np.savez_compressed(OUT / name, **pending)
            for key in pending:
                manifest[key]['shard'] = name
            shard_index += 1
            pending = {}

    with tarfile.open(fileobj=ResumingStream(links[archive]), mode='r|xz') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.h5'):
                continue
            seen += 1
            key = Path(member.name).stem
            if key not in wanted:
                continue
            temp = OUT / f'{archive}_current.h5'
            with temp.open('wb') as handle:
                source = tar.extractfile(member)
                while True:
                    block = source.read(1 << 24)
                    if not block:
                        break
                    handle.write(block)
            with h5py.File(temp, 'r') as h5:
                volume = np.asarray(h5['reconstruction_esc'], dtype=np.float32)
                acquisition = str(h5.attrs.get('acquisition', ''))
            temp.unlink()
            scale = float(np.percentile(volume, 99.5)) or 1.0
            pending[key] = np.clip(volume / scale * 255, 0, 255).astype(np.uint8)
            manifest[key] = {'slices': int(volume.shape[0]), 'shape': list(volume.shape[1:]),
                             'acquisition': acquisition, 'scale': scale, 'split': archive}
            if len(pending) == 64:
                flush()
            if len(manifest) % 50 == 0:
                print(archive, 'members', seen, 'kept', len(manifest), round(time.time() - started), flush=True)
    flush()
    (OUT / f'{archive}_manifest.json').write_text(json.dumps(manifest))
    print(archive, 'done: members', seen, 'kept', len(manifest), round(time.time() - started), flush=True)


def merge():
    wanted = {line.strip().replace('.h5', '') for line in
              (ROOT / 'data/external/fastmri_plus/knee_file_list.csv').read_text().splitlines() if line.strip()}
    manifest = {}
    for archive in ('knee_singlecoil_val', 'knee_singlecoil_train'):
        manifest.update(json.loads((OUT / f'{archive}_manifest.json').read_text()))
    missing = sorted(wanted - set(manifest))
    print('kept', len(manifest), 'of', len(wanted), 'missing', len(missing), flush=True)
    assert len(manifest) >= 0.95 * len(wanted), 'too many annotated exams missing'
    for name in ('knee.csv', 'knee_file_list.csv'):
        (OUT / name).write_bytes((ROOT / 'data/external/fastmri_plus' / name).read_bytes())
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('knee_singlecoil_*.npz'))}
    (OUT / 'manifest.json').write_text(json.dumps({'exams': manifest, 'missing': missing, 'sha256': digests}, indent=1))


if __name__ == '__main__':
    merge() if sys.argv[1] == 'merge' else build(sys.argv[1])
