"""Create a code-and-weights bundle and an offline Kaggle inference notebook."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--username', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    stage = root / 'artifacts/kaggle_dataset'
    notebook = root / 'artifacts/kaggle_notebook'
    stage.mkdir(parents=True, exist_ok=True); notebook.mkdir(parents=True, exist_ok=True)
    weights = root / 'artifacts/remote/inference_bundle'
    required = [weights / 'encoder.pt', weights / 'model2/head.pt', weights / 'model3/head.pt',
                weights / 'model5/encoder.pt', weights / 'model5/head.pt', weights / 'ensemble_config.json']
    if not all(f.exists() for f in required):
        raise RuntimeError('Trained weights are not available yet')
    if not (root / 'artifacts/remote/ENSEMBLE_VALIDATED').exists():
        raise RuntimeError('Real DICOM inference must pass before packaging')
    with zipfile.ZipFile(stage / 'rsna_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in required:
            archive.write(file, 'weights/' + str(file.relative_to(weights)))
        for file in (root / 'knee').glob('*.py'):
            archive.write(file, 'knee/' + file.name)
        for file in (root / 'third_party').rglob('*'):
            if file.is_file(): archive.write(file, str(file.relative_to(root)))
        archive.write(root / 'SOURCES.md', 'SOURCES.md')
    for wheel in (root / 'artifacts/remote/wheels').glob('*.whl'):
        shutil.copy(wheel, stage / wheel.name)
    slug = 'rsna-knee-dinov2-pilot-weights'
    metadata = {'title': 'RSNA Knee DINOv2 Pilot Weights', 'id': f'{args.username}/{slug}',
                'licenses': [{'name': 'other'}],
                'description': 'Private pilot: frozen and fine-tuned DINOv2-small MRI models. Code and model weights use Apache-2.0 terms; see bundled SOURCES.md and third-party notices for attribution. No competition pixels or reports are included.'}
    (stage / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2))
    code = '''from pathlib import Path
import subprocess, sys, zipfile, json, shutil
import pandas as pd

inputs = Path('/kaggle/input')
dataset_candidates = [inputs / 'rsna-knee-dinov2-pilot-weights']
dataset_candidates += list(inputs.glob('datasets/*/rsna-knee-dinov2-pilot-weights'))
dataset_candidates += list(inputs.glob('datasets/*/rsna-knee-dinov2-pilot-weights/*'))
dataset_candidates = [p for p in dataset_candidates if p.is_dir() and
                      ((p / 'rsna_bundle.zip').exists() or (p / 'weights/ensemble_config.json').exists()
                       or (p / 'rsna_bundle/weights/ensemble_config.json').exists())]
dataset_candidates = [p for p in dataset_candidates
                      if not any(q in p.parents for q in dataset_candidates if q != p)]
assert len(dataset_candidates) == 1, dataset_candidates
dataset = dataset_candidates[0]
bundles = list(dataset.glob('rsna_bundle.zip'))
for wheel in dataset.glob('timm-*.whl'):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheel)], check=True)
work = Path('/kaggle/working/rsna')
work.mkdir(exist_ok=True)
if bundles:
    with zipfile.ZipFile(bundles[0]) as z:
        z.extractall(work)
else:
    # Kaggle may expand uploaded archives during dataset processing.
    roots = [p for p in [dataset, dataset / 'rsna_bundle'] if (p / 'knee/infer.py').exists()]
    assert len(roots) == 1, roots
    shutil.copytree(roots[0], work, dirs_exist_ok=True)
candidates = [p for p in [inputs / 'rsna-knee-abnormality-detection',
                          inputs / 'competitions/rsna-knee-abnormality-detection']
              if (p / 'test.csv').exists() and (p / 'test_series').is_dir()]
assert len(candidates) == 1, candidates
alpha = json.loads((work / 'weights/ensemble_config.json').read_text())['fine_tuned_weight']
frames = []
for models, weight, filename in [(['model2','model3'], 1-alpha, 'frozen.csv'), (['model5'], alpha, 'tuned.csv')]:
    if weight == 0:
        continue
    output = work / filename
    subprocess.run([sys.executable, '-m', 'knee.infer', '--data', str(candidates[0]),
                    '--weights', str(work / 'weights'), '--models', *models,
                    '--output', str(output)], cwd=work, check=True)
    frames.append((pd.read_csv(output).set_index('StudyInstanceUID'), weight))
result = frames[0][0] * frames[0][1]
for frame, weight in frames[1:]:
    result += frame.loc[result.index] * weight
result.reset_index().to_csv('/kaggle/working/submission.csv', index=False)
'''
    nb = {'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
          'cells': [{'cell_type': 'markdown', 'id': 'description', 'metadata': {}, 'source': [
              '# RSNA Knee: DINOv2 pilot ensemble\n',
              'Six scan slots, neighboring-slice MRI inputs, frozen and fine-tuned encoders. The blend uses weak validation; reports are used only for training labels. Source attribution is included in the weights bundle.\n']},
                    {'cell_type': 'code', 'id': 'inference', 'metadata': {}, 'execution_count': None,
                     'outputs': [], 'source': code.splitlines(keepends=True)}]}
    (notebook / 'inference.ipynb').write_text(json.dumps(nb, indent=2))
    kernel_meta = {'id': f'{args.username}/rsna-knee-dinov2-pilot', 'title': 'RSNA Knee DINOv2 Pilot',
                   'code_file': 'inference.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                   'is_private': True, 'enable_gpu': True, 'enable_internet': False,
                   'machine_shape': 'NvidiaTeslaT4',
                   'dataset_sources': [f'{args.username}/{slug}'],
                   'competition_sources': ['rsna-knee-abnormality-detection'],
                   'kernel_sources': [], 'model_sources': []}
    (notebook / 'kernel-metadata.json').write_text(json.dumps(kernel_meta, indent=2))
    print(stage); print(notebook)


if __name__ == '__main__':
    main()
