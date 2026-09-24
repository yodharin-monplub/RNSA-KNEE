"""Build and run the self-contained model16/model20 blend as a private T4 notebook."""
import ast
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi


def model20_source(model16_source):
    changes = {
        'Model16: weak-label fine-tuned Max-Span checkpoint; same 2%-98% sampling.':
            ('Model20: weak-label fine-tuned native384 checkpoint; direct 384-pixel geometry.', 1),
        'volume = np.zeros((64, 336, 336), dtype=np.uint8)':
            ('volume = np.zeros((64, 384, 384), dtype=np.uint8)', 1),
        '(336,336), interpolation=cv2.INTER_AREA)':
            ('(384,384), interpolation=cv2.INTER_AREA)', 1),
        "filename = 'model16_best.pt'": ("filename = 'model20_best.pt'", 1),
        "inputs/'rsna-knee-model16-weights'/filename":
            ("inputs/'rsna-knee-model20-weights'/filename", 1),
        "datasets/yodharinmonplub/rsna-knee-model16-weights/":
            ("datasets/yodharinmonplub/rsna-knee-model20-weights/", 2),
    }
    result = model16_source
    for old, (new, expected) in changes.items():
        assert result.count(old) == expected, old
        result = result.replace(old, new)
    assert 'np.zeros((64, 336, 336)' not in result
    assert '(336,336), interpolation=cv2.INTER_AREA)' not in result
    return result


def main():
    assert (ROOT / 'artifacts/model16_dataset/PACKAGE_READY.json').exists()
    assert (ROOT / 'artifacts/model20_dataset/PACKAGE_READY.json').exists()
    first = (ROOT / 'model16/inference.py').read_text()
    second = model20_source(first)
    ast.parse(first); ast.parse(second)
    wrapper = f'''"""Fixed equal model16/model20 probability blend; see model21/readme.txt."""
import ast
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
MODEL16_CODE = {first!r}
MODEL20_CODE = {second!r}

def main():
    started = time.time()
    out = Path('/kaggle/working')
    frames = []
    for name, code in [('model16', MODEL16_CODE), ('model20', MODEL20_CODE)]:
        ast.parse(code)
        script = out / f'{{name}}_component.py'
        script.write_text(code)
        subprocess.run([sys.executable, str(script)], check=True)
        destination = out / f'{{name}}_predictions.csv'
        (out / 'submission.csv').replace(destination)
        frames.append(pd.read_csv(destination).set_index('StudyInstanceUID'))
    first_frame = frames[0]
    arrays = []
    for frame in frames:
        assert frame.index.is_unique and set(frame.index) == set(first_frame.index)
        assert list(frame.columns) == list(first_frame.columns)
        values = frame.loc[first_frame.index].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        arrays.append(values)
    prediction = (arrays[0] + arrays[1]) / 2
    assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
    pd.DataFrame(prediction, index=first_frame.index,
                 columns=first_frame.columns).to_csv(out / 'submission.csv')
    runtime = {{'studies': len(first_frame), 'elapsed_seconds': time.time() - started,
                'weights': {{'model16': 0.5, 'model20': 0.5}}}}
    (out / 'ensemble_runtime.json').write_text(json.dumps(runtime, indent=2))
    print(json.dumps(runtime), flush=True)

if __name__ == '__main__':
    main()
'''
    ast.parse(wrapper)
    module = ast.parse(wrapper)
    constants = {node.targets[0].id: node.value.value for node in module.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)
                 and isinstance(node.value.value, str)}
    assert constants['MODEL16_CODE'] == first
    assert constants['MODEL20_CODE'] == second

    api = KaggleApi(); api.authenticate(); user = api.get_config_value('username')
    out = ROOT / 'artifacts/model21_notebook'; out.mkdir(parents=True, exist_ok=True)
    (out / 'inference.py').write_text(wrapper)
    metadata = {
        'id': f'{user}/rsna-knee-model21-dual-fine-tuned-blend',
        'title': 'RSNA Knee Model21 Dual Fine Tuned Blend',
        'code_file': 'inference.py', 'language': 'python', 'kernel_type': 'script',
        'is_private': True, 'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4',
        'enable_internet': False,
        'dataset_sources': [f'{user}/rsna-knee-model16-weights',
                            f'{user}/rsna-knee-model20-weights'],
        'competition_sources': ['rsna-knee-abnormality-detection'],
        'kernel_sources': [], 'model_sources': [],
    }
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    result = api.kernels_push(str(out), acc='NvidiaTeslaT4')
    (out / 'run_response.json').write_text(str(result))
    print(str(result))


if __name__ == '__main__':
    main()
