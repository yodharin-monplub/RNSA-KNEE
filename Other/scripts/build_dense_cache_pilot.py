"""CPU cache pilot using model12's exact DICOM preprocessing.
Geometry credit: Johnathan Wagner / Dread Development, Apache-2.0 implementation.
Competition-derived pixels retain competition data restrictions.
"""
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixels import apply_modality_lut
cv2.setNumThreads(1)
UID = 'StudyInstanceUID'
CACHE_SIZE = 336
CACHE_SPAN = (.02, .98)

SLOTS = [('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12),
         ('Coronal', 0, 8), ('Axial', -1, 12)]

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

def ordered_files(directory):
    records = []
    for path in directory.glob('*.dcm'):
        header = pydicom.dcmread(path, stop_before_pixels=True)
        orientation = getattr(header, 'ImageOrientationPatient', None)
        position = getattr(header, 'ImagePositionPatient', None)
        if orientation is not None and position is not None and len(orientation) == 6:
            normal = np.cross(np.asarray(orientation[:3], float), np.asarray(orientation[3:], float))
            order = float(np.dot(np.asarray(position, float), normal))
        else:
            order = float(getattr(header, 'InstanceNumber', 0) or 0)
        spacing = getattr(header, 'PixelSpacing', None)
        records.append((order, path, float(spacing[0]) if spacing is not None else .5))
    records.sort(key=lambda x: x[0])
    return records

def read_study(uid, rows, series_root):
    volume = np.zeros((64, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
    used, offset, failed = set(), 0, 0
    for plane, fluid, count in SLOTS:
        candidates = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
        preferred = [r for r in candidates if int(r['Fluid_Sensitive']) == fluid] if fluid in (0,1) else []
        chosen = (preferred or candidates)
        if not chosen:
            offset += count
            continue
        row = chosen[0]; used.add(row['SeriesInstanceUID'])
        records = ordered_files(series_root / uid / row['SeriesInstanceUID'])
        if not records:
            raise FileNotFoundError('Declared series has no DICOM files')
        n = len(records); lo = int(n * CACHE_SPAN[0]); hi = max(lo, int(n * CACHE_SPAN[1])-1)
        indices = np.rint(np.linspace(lo, hi, count)).astype(int) if n > 1 else np.zeros(count, dtype=int)
        selected = []
        for i in indices:
            _, path, spacing = records[min(i, n-1)]
            try:
                ds = pydicom.dcmread(path)
                array = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
                if array.ndim != 2 or not np.isfinite(array).all():
                    raise ValueError('Invalid decoded MRI pixels')
                if str(getattr(ds, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
                    array = array.max() - array
                selected.append((array, spacing))
            except (ValueError, RuntimeError, NotImplementedError):
                selected.append((None, spacing)); failed += 1
        valid = [a for a, _ in selected if a is not None]
        if not valid:
            raise ValueError('No decodable slices in selected series')
        low, high = np.percentile(np.concatenate([a.ravel() for a in valid]), [2, 98])
        median_spacing = float(np.median([r[2] for r in records]))
        for array, spacing in selected:
            if array is not None:
                array = np.clip((array-low)/(high-low+1e-6), 0, 1)
                spacing = spacing if spacing > 0 else median_spacing
                h, w = array.shape
                size = min(int(round(140/max(spacing, 1e-3))), h, w)
                y, x = (h-size)//2, (w-size)//2
                crop = cv2.resize(array[y:y+size, x:x+size], (CACHE_SIZE,CACHE_SIZE), interpolation=cv2.INTER_AREA)
                volume[offset] = (crop * 255).astype(np.uint8)
            offset += 1
    mask = volume.reshape(64,-1).sum(1) > 0
    if not mask.any():
        raise ValueError('Study has no usable images')
    return volume, mask, failed

def main():
    started=time.time()
    roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
    root=next(p for p in roots if (p/'train.csv').exists())
    train=pd.read_csv(root/'train.csv')
    gold=train.dropna(subset=LABELS)
    weak=train[~train[UID].isin(gold[UID])].head(6)
    ids=gold[UID].tolist()+weak[UID].tolist()
    assert len(ids)==64 and len(set(ids))==64
    series=pd.read_csv(root/'train_series.csv')
    groups={uid:f.to_dict('records') for uid,f in series[series[UID].isin(ids)].groupby(UID)}
    out=Path('/kaggle/working')
    def read(uid):
        volume,mask,failed=read_study(uid,groups[uid],root/'train_series')
        assert volume.shape==(64,CACHE_SIZE,CACHE_SIZE) and volume.dtype==np.uint8
        return uid,volume,mask,failed
    manifest={'kind':'pilot','studies':64,'gold_studies':58,'geometry':{'slots':SLOTS,'span':list(CACHE_SPAN),'crop_mm':140,'size':CACHE_SIZE},'shards':[]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,len(ids),16):
            rows=list(pool.map(read,ids[start:start+16]))
            filename=f'pilot_{start//16:02d}.npz'
            path=out/filename
            np.savez_compressed(path,ids=np.asarray([r[0] for r in rows]),volumes=np.stack([r[1] for r in rows]),masks=np.stack([r[2] for r in rows]))
            with np.load(path,allow_pickle=False) as cached:
                assert cached['ids'].tolist()==[r[0] for r in rows]
                assert np.array_equal(cached['volumes'],np.stack([r[1] for r in rows]))
                assert np.array_equal(cached['masks'],np.stack([r[2] for r in rows]))
            digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
            manifest['shards'].append({'file':filename,'ids':[r[0] for r in rows],'bytes':path.stat().st_size,'sha256':digest,'failed_slices':sum(r[3] for r in rows)})
            print('Cached',start+len(rows),'/',len(ids),'seconds',round(time.time()-started,1),flush=True)
    manifest['elapsed_seconds']=time.time()-started
    manifest['compressed_bytes']=sum(s['bytes'] for s in manifest['shards'])
    manifest['estimated_full_compressed_bytes']=manifest['compressed_bytes']/64*len(train)
    manifest['note']='Pilot storage estimate from 58 gold + 6 weak cases; not a representative guarantee.'
    (out/'cache_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='shards'}),flush=True)


if __name__=='__main__':
    main()
