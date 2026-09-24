"""Launch a two-case GPU smoke test for model22 using the 6%-94% pilot cache."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    files = {name:(ROOT / name).read_text() for name in [
        'scripts/train_model16.py','knee/dense.py','knee/data.py','cloud/prepare_dense_cache.py',
        'model12/inference.py','model22/config.json']}
    source = '''"""Private model22 GPU smoke test on two weak training cases."""\nimport json,subprocess,sys\nfrom pathlib import Path\nROOT=Path('/kaggle/working/rsna_model22_smoke');ROOT.mkdir(exist_ok=True)\n'''
    source += 'FILES = ' + repr(files) + r'''
for name,content in FILES.items():
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
for package in ['knee','cloud','model12','model22']:(ROOT/package/'__init__.py').write_text('')
inputs=Path('/kaggle/input')
def locate(slug,owner,kind,filename):
    candidates=[inputs/slug,inputs/kind/owner/slug]
    matches=[p for p in candidates if (p/filename).exists()]
    assert len(matches)==1,(slug,matches);return matches[0]
comp=next(p for p in [inputs/'competitions/rsna-knee-abnormality-detection',inputs/'rsna-knee-abnormality-detection'] if (p/'train.csv').exists())
weak=locate('rsna-knee-llm-report-labels','stevenleehans','datasets','llm_labels_full.csv')
cache=locate('rsna-knee-dense-cache-336-6-94-pilot','yodharinmonplub','notebooks','cache_manifest.json')
weights=locate('raptor-knee-finespacing','dreaddevelopment','datasets','raptor_ft_coatnet_v9_full.pt')
for origin,dest in [(comp/'train.csv','data/train.csv'),(comp/'sample_submission.csv','data/sample_submission.csv'),(weak/'llm_labels_full.csv','data/weak_labels/llm_labels_full.csv'),(weak/'llm_labels_v4_blend.csv','data/weak_labels_v4/llm_labels_v4_blend.csv')]:
    target=ROOT/dest;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(origin)
sys.path.insert(0,str(ROOT))
from knee.data import make_partitions
make_partitions(ROOT/'data',ROOT/'artifacts/partitions')
subprocess.run([sys.executable,str(ROOT/'cloud/prepare_dense_cache.py'),'--manifest',str(cache/'cache_manifest.json'),'--train',str(ROOT/'data/train.csv'),'--destination',str(ROOT/'dense_cache'),'--pilot','--size','336','--span-start','.06','--span-end','.94'],check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/train_model16.py'),'--cache',str(ROOT/'dense_cache'),'--checkpoint',str(weights/'raptor_ft_coatnet_v9_full.pt'),'--output','/kaggle/working/model22_smoke','--smoke','--config',str(ROOT/'model22/config.json'),'--cache-size','336','--span-start','.06','--span-end','.94'],check=True)
print(Path('/kaggle/working/model22_smoke/SMOKE_PASS.json').read_text(),flush=True)
'''
    api = KaggleApi(); api.authenticate(); user = api.get_config_value('username')
    out = ROOT / 'artifacts/model22_smoke_notebook'; out.mkdir(parents=True, exist_ok=True)
    (out / 'inference.py').write_text(source)
    meta = {'id':f'{user}/rsna-knee-model22-smoke','title':'RSNA Knee Model22 Smoke',
            'code_file':'inference.py','language':'python','kernel_type':'script','is_private':True,
            'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':False,
            'dataset_sources':['dreaddevelopment/raptor-knee-finespacing','stevenleehans/rsna-knee-llm-report-labels'],
            'competition_sources':['rsna-knee-abnormality-detection'],
            'kernel_sources':[f'{user}/rsna-knee-dense-cache-336-6-94-pilot'],'model_sources':[]}
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    result = api.kernels_push(str(out), acc='NvidiaTeslaT4')
    (out / 'run_response.json').write_text(str(result)); print(str(result))


if __name__ == '__main__':
    main()
