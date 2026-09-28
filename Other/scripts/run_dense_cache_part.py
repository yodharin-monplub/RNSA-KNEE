"""Run the attributed public CoAtNet reproduction as a private T4 notebook."""
import argparse
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env');os.environ.setdefault('KAGGLE_API_TOKEN',os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR']=str(ROOT/'.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

parser=argparse.ArgumentParser();parser.add_argument('--part',type=int,choices=range(4),required=True);parser.add_argument('--size',type=int,choices=[256,336,384],default=336);parser.add_argument('--span',choices=['2-98','6-94'],default='2-98');parser.add_argument('--slots',type=int,choices=[64,44,96,128],default=64);parser.add_argument('--dry',action='store_true');args=parser.parse_args()
api=KaggleApi();api.authenticate();user=api.get_config_value('username')
span_tag='' if args.span=='2-98' else '_6_94'
span_slug='' if args.span=='2-98' else ('-336-6-94' if args.size==336 else '-6-94')
tag=('' if args.size==336 else f'_{args.size}')+span_tag+('' if args.slots==64 else f'_s{args.slots}')
slug=('' if args.size==336 else f'-{args.size}')+span_slug+('' if args.slots==64 else f'-s{args.slots}')
out=ROOT/f'artifacts/dense_cache{tag}_part{args.part}_notebook';out.mkdir(parents=True,exist_ok=True)
source=(ROOT/'scripts/build_dense_cache_part.py').read_text().replace('CACHE_PART = 0',f'CACHE_PART = {args.part}').replace('CACHE_SIZE = 336',f'CACHE_SIZE = {args.size}')
if args.span=='6-94':source=source.replace('CACHE_SPAN = (.02, .98)','CACHE_SPAN = (.06, .94)')
SLOTS64_SRC="SLOTS = [('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12),\n         ('Coronal', 0, 8), ('Axial', -1, 12)]"
SLOTS44_SRC="SLOTS = [('Sagittal', 1, 12), ('Sagittal', 0, 10), ('Coronal', 1, 8),\n         ('Coronal', 0, 6), ('Axial', -1, 8)]"
# 128 frames: 24-32 slices per series (forum: 24 -> 32 cached slices per series beat any label work; Speedy's Raptor
# stage reads ~96). Sagittal fluid keeps the largest share, as in the 64-frame layout.
SLOTS128_SRC="SLOTS = [('Sagittal', 1, 32), ('Sagittal', 0, 24), ('Coronal', 1, 24),\n         ('Coronal', 0, 24), ('Axial', -1, 24)]"
if args.slots==44:
    assert source.count(SLOTS64_SRC)==1;source=source.replace(SLOTS64_SRC,SLOTS44_SRC)
if args.slots==128:
    assert source.count(SLOTS64_SRC)==1;source=source.replace(SLOTS64_SRC,SLOTS128_SRC)
# 96 frames at 336 px for the CoAtNet-2 384 recipe (the 128-frame 256 px cache helped EfficientNet-B0 by +0.002 LB)
SLOTS96_SRC="SLOTS = [('Sagittal', 1, 24), ('Sagittal', 0, 18), ('Coronal', 1, 18),\n         ('Coronal', 0, 18), ('Axial', -1, 18)]"
if args.slots==96:
    assert source.count(SLOTS64_SRC)==1;source=source.replace(SLOTS64_SRC,SLOTS96_SRC)
(out/'inference.py').write_text(source)
title=(f'RSNA Knee Dense Cache {args.size} {args.span} Part{args.part}' if args.span!='2-98' else f'RSNA Knee Dense Cache {args.size} Part{args.part}')+('' if args.slots==64 else f' S{args.slots}')
meta={'id':f'{user}/rsna-knee-dense-cache{slug}-part{args.part}','title':title,
      'code_file':'inference.py','language':'python','kernel_type':'script','is_private':True,
      'enable_gpu':False,'enable_internet':False,
      'dataset_sources':[],
      'competition_sources':['rsna-knee-abnormality-detection'],'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
import sys
if '--dry' in sys.argv: print('dry: built', out); raise SystemExit
result=api.kernels_push(str(out))
(out/'run_response.json').write_text(str(result));print(str(result))
