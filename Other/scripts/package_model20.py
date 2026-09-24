"""Stage completed model20 weights for a private Kaggle dataset; no upload."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'artifacts/remote_model20/model20'
    if not (source / 'TRAINING_COMPLETE').is_file():
        raise RuntimeError('Full training and artifact backup must complete before packaging')
    metrics = json.loads((source / 'metrics.json').read_text())
    history = json.loads((source / 'history.json').read_text())
    assert metrics['selected_epoch'] > 0, 'Epoch zero retained: use existing model15 instead'
    assert metrics['weak_val_bce'] < history[0]['weak_val_bce']
    checkpoint = source / 'best.pt'
    assert checkpoint.stat().st_size > 1_000_000
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    output = ROOT / 'artifacts/model20_dataset'
    output.mkdir(exist_ok=True)
    (output / 'PACKAGE_READY.json').unlink(missing_ok=True)
    shutil.copy2(checkpoint, output / 'model20_best.pt')
    for path in [ROOT / 'SOURCES.md', ROOT / 'model20/readme.txt']:
        shutil.copy2(path, output / path.name)
    metadata = {
        'title': 'RSNA Knee Model20 Weights',
        'id': 'yodharinmonplub/rsna-knee-model20-weights',
        'licenses': [{'name': 'other'}],
        'description': ('Private fine-tuned native384 CoAtNet checkpoint. Initial model and geometry '
                        'by Johnathan Wagner / Dread Development. No MRI pixels, reports, or patient '
                        'identifiers included. See SOURCES.md.'),
    }
    (output / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2))
    (output / 'PACKAGE_READY.json').write_text(json.dumps({
        'sha256': digest,
        'source_checkpoint_sha256': '5add8816a47e040efca89bb44480e1cbd01f92db1253831ff7ceb6c65ee02b1d',
        'metrics': metrics,
    }, indent=2))
    print('Private model20 weight package ready; upload and GPU inference validation still required.')


if __name__ == '__main__':
    main()

