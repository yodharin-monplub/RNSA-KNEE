import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from knee.data import PixelCache, UID
from knee.extract import make_encoder
from knee.models import StudyHead, confidence_bce
from knee.train_heads import score


class TrainingPixels(Dataset):
    def __init__(self, cache, indices, targets, confidence):
        self.cache, self.indices = cache, np.asarray(indices)
        self.targets, self.confidence = targets, confidence

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        pixels, mask = self.cache[index]
        return pixels, mask, self.targets[index], self.confidence[index]


class TunableStudy(nn.Module):
    def __init__(self, cfg, checkpoint):
        super().__init__()
        self.encoder = make_encoder(cfg['backbone'], pretrained=False)
        self.encoder.load_state_dict(torch.load('artifacts/features/encoder.pt', map_location='cpu', weights_only=True))
        self.encoder.requires_grad_(False)
        for block in self.encoder.blocks[-cfg['unfrozen_blocks']:]:
            block.requires_grad_(True)
        self.encoder.norm.requires_grad_(True)
        self.head = StudyHead(checkpoint['dim'], cfg['head'], cfg['dropout'], len(checkpoint['labels']))
        self.head.load_state_dict(checkpoint['state_dict'])
        self.register_buffer('feature_mean', checkpoint['mean'])
        self.register_buffer('feature_std', checkpoint['std'])
        self.register_buffer('image_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, pixels, mask):
        batch = len(pixels)
        x = pixels.reshape(batch, 18, 3, 224, 224).float() / 255
        if self.training:
            flips = torch.rand(batch, device=x.device) < .5
            x[flips] = x[flips].flip(-1)
            gamma = torch.empty(batch, 1, 1, 1, 1, device=x.device).uniform_(.9, 1.1)
            x = x.pow(gamma)
        x = x.reshape(batch * 18, 3, 224, 224)
        keep = mask[:, :, None].expand(-1, -1, 3).reshape(-1)
        selected = (x[keep] - self.image_mean) / self.image_std
        f = self.encoder(selected)
        # index_copy keeps gradients to the selected features while masking gaps.
        all_features = torch.zeros(batch * 18, self.encoder.num_features, dtype=f.dtype, device=f.device)
        all_features = all_features.index_copy(0, torch.where(keep)[0], f)
        all_features = all_features.reshape(batch, 6, 3, -1)
        normalized = ((all_features.float() - self.feature_mean) / self.feature_std) * mask[:, :, None, None]
        return self.head(normalized, mask)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    losses = torch.zeros(12, device=device)
    counts = torch.zeros(12, device=device)
    predictions = []
    for pixels, mask, targets, weights in loader:
        pixels, mask = pixels.to(device), mask.to(device)
        targets, weights = targets.to(device), weights.to(device)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(pixels, mask)
        loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), targets, reduction='none')
        losses += (loss * weights).sum(0); counts += weights.sum(0)
        predictions.append(logits.float().sigmoid().cpu().numpy())
    valid = counts > 0
    return float((losses[valid] / counts[valid]).mean()), np.concatenate(predictions)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', default='model5/config.json')
    args = p.parse_args(); cfg = json.loads(Path(args.config).read_text())
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('GPU is required')
    device = torch.device('cuda')
    parts = pd.read_csv('artifacts/partitions/partitions.csv')
    targets = np.load('artifacts/partitions/targets.npz')
    cache = PixelCache('data/pixels', parts[UID])
    indices = {split: np.flatnonzero(parts.split.eq(split)) for split in ['train', 'weak_val', 'gold']}
    loaders = {split: DataLoader(TrainingPixels(cache, index, targets['weak'], targets['confidence']),
                                batch_size=cfg['batch_size'], shuffle=split=='train',
                                num_workers=4, pin_memory=True, persistent_workers=True)
               for split, index in indices.items()}
    source = torch.load('artifacts/model2/head.pt', map_location='cpu', weights_only=True)
    model = TunableStudy(cfg, source).to(device)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': cfg['encoder_learning_rate']},
        {'params': model.head.parameters(), 'lr': cfg['head_learning_rate']}], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs'])
    out = Path('artifacts/model5'); out.mkdir(parents=True, exist_ok=True)
    best_loss, _ = evaluate(model, loaders['weak_val'], device)
    reference_loss = json.loads(Path('artifacts/model2/metrics.json').read_text())['weak_val_bce']
    if abs(best_loss - reference_loss) > 0.001:
        raise RuntimeError(f'Pixel/feature pipeline mismatch: {best_loss} versus {reference_loss}')
    best_epoch = 0
    history = [{'epoch': 0, 'weak_val_bce': best_loss}]
    print(json.dumps(history[-1]), flush=True)

    def save_best():
        torch.save({k:v.cpu() for k,v in model.encoder.state_dict().items()}, out / 'encoder.pt')
        checkpoint = dict(source, config=cfg, state_dict={k:v.cpu() for k,v in model.head.state_dict().items()},
                          best_epoch=best_epoch)
        torch.save(checkpoint, out / 'head.pt')

    save_best(); started = time.time()
    for epoch in range(1, cfg['epochs']+1):
        model.train()
        for step, (pixels, mask, y, w) in enumerate(loaders['train']):
            pixels, mask, y, w = [t.to(device, non_blocking=True) for t in [pixels, mask, y, w]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(pixels, mask)
                loss = confidence_bce(logits.float(), y, w)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
            scaler.step(optimizer); scaler.update()
            if step % 100 == 0:
                print(json.dumps({'epoch': epoch, 'step': step, 'loss': float(loss.detach()),
                                  'seconds': round(time.time()-started, 1)}), flush=True)
        scheduler.step()
        value, _ = evaluate(model, loaders['weak_val'], device)
        history.append({'epoch': epoch, 'weak_val_bce': value})
        if value < best_loss:
            best_loss, best_epoch = value, epoch
            save_best()
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(history[-1]), flush=True)
    model.encoder.load_state_dict(torch.load(out / 'encoder.pt', map_location=device, weights_only=True))
    best_head = torch.load(out / 'head.pt', map_location=device, weights_only=True)
    model.head.load_state_dict(best_head['state_dict'])
    _, pred = evaluate(model, loaders['gold'], device)
    _, val_pred = evaluate(model, loaders['weak_val'], device)
    metrics = score(targets['gold'][indices['gold']], pred)
    metrics.update({'model': 'model5', 'best_epoch': best_epoch, 'weak_val_bce': best_loss,
                    'training_seconds': time.time()-started, 'gold_n': len(indices['gold']),
                    'leaderboard_score': None})
    for split, prediction, filename in [('gold',pred,'gold_predictions.csv'), ('weak_val',val_pred,'weak_val_predictions.csv')]:
        frame = pd.DataFrame(prediction, columns=source['labels'])
        frame.insert(0, UID, parts.iloc[indices[split]][UID].to_numpy())
        frame.to_csv(out / filename, index=False)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
