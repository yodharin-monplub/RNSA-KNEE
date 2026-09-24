"""Run the attributed public CoAtNet reproduction as a private T4 notebook."""
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env');os.environ.setdefault('KAGGLE_API_TOKEN',os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR']=str(ROOT/'.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

api=KaggleApi();api.authenticate();user=api.get_config_value('username')
out=ROOT/'artifacts/model13_notebook';out.mkdir(parents=True,exist_ok=True)
shutil.copy(ROOT/'model13/inference.py',out/'inference.py')
meta={'id':f'{user}/rsna-knee-model13-dinov3-folds','title':'RSNA Knee Model13 DINOv3 Folds',
      'code_file':'inference.py','language':'python','kernel_type':'script','is_private':True,
      'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':False,
      'dataset_sources':['mattiaangeli/knee-mri-fold-weights'],
      'competition_sources':['rsna-knee-abnormality-detection'],'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
result=api.kernels_push(str(out),acc='NvidiaTeslaT4')
(out/'run_response.json').write_text(str(result));print(str(result))
