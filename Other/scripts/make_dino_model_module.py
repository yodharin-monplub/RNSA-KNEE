"""Generate knee/dino_model.py from the public notebook's own member architecture cells.

SlotHead, Model and the fingerprint helper are copied verbatim from
mattiaangeli/bend-the-knee-to-the-dinosaurs (cells 8, 9 and the fingerprint part of cell 11) so that
fine-tuned members compute exactly what the released ones do. Only build_model is replaced: it
constructs the DINOv2-small backbone from the pinned config (third_party/dinov2/config.json) instead
of downloading pretrained weights, because every member checkpoint already carries full weights.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / ('artifacts/sources/public_notebooks/mattiaangeli__bend-the-knee-to-the-dinosaurs/'
                   'bend-the-knee-to-the-dinosaurs.ipynb')

cells = [''.join(c['source']) for c in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']]
slot_head, model_cell, cell11 = cells[8], cells[9], cells[11]
fingerprint = cell11[cell11.index('def fingerprint('):cell11.index('def check_fingerprint(')]
assert 'class SlotHead' in slot_head and 'class Model' in model_cell and 'torch.randint' in fingerprint

header = '''"""DINOv2 member architecture, copied from the public notebook (see scripts/make_dino_model_module.py).

Source: https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs (Apache-2.0),
model by Pilkwang / Sofia Anjenje et al.; see SOURCES.md.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker\\'s", 'Contusion', 'Fracture']
SLOTS = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True),
         ('AX_FLUID_FS', 'Axial', True, True), ('SAG_FLUID_NOFS', 'Sagittal', True, False),
         ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
N_SLOT = len(SLOTS)
GROUP = 3
SEED = 2026
POOL_PARTS = {'cls_mean': 2, 'cls_mean_focal': 3}
SLOT_PRIOR_TABLE = {'ACL': (0, 3, 5), 'MCL': (1, 4), 'Medial Meniscus': (0, 1, 3, 4),
                    'Lateral Meniscus': (0, 1, 3, 4), 'Medial OA': (1, 4, 5), 'Lateral OA': (1, 4, 5),
                    'PF OA': (0, 2, 5), 'Effusion': (0, 2), 'Synovitis': (0, 2), "Baker\\'s": (0,),
                    'Contusion': (0, 1, 2), 'Fracture': (0, 1, 2, 4, 5)}
SLOT_PRIOR_STRENGTH = 0.55


def log(message):
    print(message, flush=True)


'''

build_model = '''

def build_model(unfreeze_last, source=None, variant='small', pool='cls_mean', prior=False):
    """As the notebook's build_model, but the backbone is built from the pinned config (no download)."""
    from transformers import Dinov2Config, Dinov2Model
    assert variant == 'small'
    config = Dinov2Config(**json.loads((ROOT / 'third_party/dinov2/config.json').read_text()))
    bb = Dinov2Model(config)
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    return Model(bb, dim, pool=pool, prior=prior)
'''

module = header + slot_head + '\n\n' + model_cell + '\n\n' + build_model + '\n\n' + fingerprint
compile(module, 'dino_model', 'exec')
(ROOT / 'knee/dino_model.py').write_text(module)
print('wrote knee/dino_model.py', len(module), 'chars')
