"""Launch cached-versus-raw parity validation for the model10 6%-94% cache."""
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
    api = KaggleApi(); api.authenticate(); user = api.get_config_value('username')
    source = (ROOT / 'model10/inference.py').read_text()
    source = source.rsplit("if __name__ == '__main__':", 1)[0]
    reference = (ROOT / 'artifacts/coatnet_gold_notebook/output/model10_predictions.csv').read_text()
    validator = r'''
def validate_cache():
    import hashlib
    import io
    started=time.time();torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');assert device.type=='cuda'
    inputs=Path('/kaggle/input')
    roots=[p for p in [inputs/'rsna-knee-dense-cache-336-6-94-pilot',inputs/'notebooks/yodharinmonplub/rsna-knee-dense-cache-336-6-94-pilot'] if (p/'cache_manifest.json').exists()]
    assert len(roots)==1,roots
    cache=roots[0];manifest=json.loads((cache/'cache_manifest.json').read_text())
    assert manifest['studies']==64 and manifest['gold_studies']==58
    assert manifest['geometry']=={'slots':[list(x) for x in SLOTS],'span':[.06,.94],'crop_mm':140,'size':336}
    candidates=[inputs/'raptor-knee-finespacing/raptor_ft_coatnet_v9_full.pt',inputs/'datasets/dreaddevelopment/raptor-knee-finespacing/raptor_ft_coatnet_v9_full.pt']
    files=[p for p in candidates if p.is_file()];assert len(files)==1,files
    ck=torch.load(files[0],map_location='cpu',weights_only=True)
    assert list(ck.get('lab',LABELS))==LABELS and int(ck['res'])==384
    model=Classifier(ck['arch']);model.load_state_dict(ck['model'],strict=True);model=model.to(device).eval();del ck
    reference=pd.read_csv(io.StringIO(REFERENCE_CSV)).set_index(UID);predictions={}
    for shard in manifest['shards']:
        path=cache/shard['file']
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==shard['sha256']
        with np.load(path,allow_pickle=False) as archive:
            ids=archive['ids'].tolist();volumes=archive['volumes'];masks=archive['masks']
        assert ids==shard['ids'] and volumes.shape==(len(ids),64,336,336)
        for i,uid in enumerate(ids):
            if uid in reference.index:predictions[uid]=predict(model,windows(volumes[i],masks[i],384),device)
        print('Validated cached cases',len(predictions),flush=True)
    assert set(predictions)==set(reference.index)
    frame=pd.DataFrame.from_dict(predictions,orient='index',columns=LABELS).loc[reference.index]
    difference=float(np.max(np.abs(frame.to_numpy()-reference[LABELS].to_numpy())))
    assert difference<.001,difference
    frame.index.name=UID;frame.to_csv('/kaggle/working/cached_gold_predictions.csv')
    result={'studies':len(frame),'max_abs_difference_vs_raw':difference,'tolerance':.001,'elapsed_seconds':time.time()-started,'status':'PASS'}
    Path('/kaggle/working/cache_inference_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':validate_cache()
'''
    code = source + '\nREFERENCE_CSV = ' + repr(reference) + '\n' + validator
    out = ROOT / 'artifacts/dense694_inference_validation_notebook'; out.mkdir(parents=True, exist_ok=True)
    (out / 'inference.py').write_text(code)
    meta = {'id':f'{user}/rsna-knee-dense-6-94-inference-validation','title':'RSNA Knee Dense 6-94 Inference Validation',
            'code_file':'inference.py','language':'python','kernel_type':'script','is_private':True,
            'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':False,
            'dataset_sources':['dreaddevelopment/raptor-knee-finespacing'],
            'competition_sources':['rsna-knee-abnormality-detection'],
            'kernel_sources':[f'{user}/rsna-knee-dense-cache-336-6-94-pilot'],'model_sources':[]}
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    result = api.kernels_push(str(out), acc='NvidiaTeslaT4')
    (out / 'run_response.json').write_text(str(result)); print(str(result))


if __name__ == '__main__':
    main()
