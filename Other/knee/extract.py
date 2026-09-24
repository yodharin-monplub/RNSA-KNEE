import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from knee.data import PixelCache, UID


def make_encoder(name, pretrained=True):
    return timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=224)


@torch.inference_mode()
def encode_pixels(model, pixels, mask, device, microbatch=96):
    # Nine retained slices are three consecutive 2.5D channel groups.
    batch = pixels.shape[0]
    x = pixels.reshape(batch * 18, 3, 224, 224)
    keep = mask[:, :, None].expand(-1, -1, 3).reshape(-1).bool()
    kept_indices = torch.where(keep)[0]
    result = torch.zeros(batch * 18, model.num_features, dtype=torch.float32, device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for start in range(0, len(kept_indices), microbatch):
        indices = kept_indices[start:start + microbatch]
        images = x[indices].to(device, non_blocking=True).float().div_(255)
        images = (images - mean) / std
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=torch.float16):
            features = model(images)
        result[indices.to(device)] = features.float()
    return result.reshape(batch, 6, 3, -1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default='data/pixels')
    parser.add_argument('--out', default='artifacts/features')
    parser.add_argument('--backbone', default='vit_small_patch14_dinov2.lvd142m')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Feature extraction is intended for the rented GPU')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv('artifacts/partitions/partitions.csv')[UID]
    dataset = PixelCache(args.cache, ids)
    model = make_encoder(args.backbone).to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    features = np.zeros((len(dataset), 6, 3, model.num_features), dtype=np.float32)
    masks = np.zeros((len(dataset), 6), dtype=bool)
    started, offset = time.time(), 0
    for pixels, mask in loader:
        n = len(pixels)
        features[offset:offset+n] = encode_pixels(model, pixels, mask, device)
        masks[offset:offset+n] = mask.numpy()
        offset += n
        if offset % 128 == 0 or offset == len(dataset):
            print(json.dumps({'encoded': offset, 'total': len(dataset),
                              'seconds': round(time.time()-started, 1)}), flush=True)
    assert np.isfinite(features).all()
    np.save(out / 'embeddings.npy', features)
    np.save(out / 'mask.npy', masks)
    ids.to_frame().to_csv(out / 'ids.csv', index=False)
    torch.save(model.cpu().state_dict(), out / 'encoder.pt')
    metadata = {'backbone': args.backbone, 'shape': list(features.shape),
                'seconds': time.time()-started, 'torch': torch.__version__,
                'timm': timm.__version__, 'gpu': torch.cuda.get_device_name()}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print('FEATURES_READY', flush=True)


if __name__ == '__main__':
    main()
