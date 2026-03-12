"""
utils/trainer.py — Generic training loop for any model.
Returns best model state and training history.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg: dict,
    device: torch.device,
    name: str = 'Model'
) -> Tuple[dict, Dict]:
    """
    Train model with early stopping. Returns (best_state_dict, history).

    history = {'train_loss': [...], 'val_loss': [...], 'val_da': [...]}
    """
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler     = GradScaler()

    best_val_da   = 0.0
    best_state    = None
    patience_ctr  = 0
    history       = {'train_loss': [], 'val_loss': [], 'val_da': []}

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'─'*60}")
    print(f"  Training: {name} | Params: {n_params:,}")
    print(f"{'─'*60}")

    for epoch in range(1, cfg['epochs'] + 1):

        # ── Train ────────────────────────────────────────────
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            try:
                with autocast():
                    logits = model(xb)
                    loss   = criterion(logits, yb)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                tr_losses.append(loss.item())
            except Exception:
                continue

        scheduler.step()

        if not tr_losses:
            continue

        # ── Validate ─────────────────────────────────────────
        model.eval()
        val_losses, preds_all, labels_all = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                try:
                    with autocast():
                        logits = model(xb)
                        loss   = criterion(logits, yb)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_losses.append(loss.item())
                    preds  = logits.argmax(dim=1)
                    preds_all.extend(preds.cpu().numpy())
                    labels_all.extend(yb.cpu().numpy())
                except Exception:
                    continue

        tr_loss  = float(np.mean(tr_losses))  if tr_losses  else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_da   = float(np.mean(np.array(preds_all) == np.array(labels_all))) * 100 \
                   if preds_all else 50.0

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['val_da'].append(val_da)

        # ── Early stopping ───────────────────────────────────
        if val_da > best_val_da:
            best_val_da  = val_da
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or patience_ctr == 0:
            print(f"  E{epoch:>3} | Loss {tr_loss:.4f}/{val_loss:.4f} | "
                  f"Val DA {val_da:.1f}% | best {best_val_da:.1f}%")

        if patience_ctr >= cfg['patience']:
            print(f"  Early stop @ epoch {epoch} (best: {epoch - patience_ctr})")
            break

    print(f"  ✓ Best Val DA: {best_val_da:.1f}%")
    return best_state, history


@torch.no_grad()
def predict(model: nn.Module, dataset, device: torch.device,
            batch_size: int = 512, noise: float = 0.0):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        try:
            if noise > 0:
                xb = xb + torch.randn_like(xb) * noise
            logits = model(xb)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(yb.numpy())
        except Exception:
            continue

    probs_arr  = np.array(all_probs,  dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)
    return probs_arr, labels_arr


@torch.no_grad()
def ensemble_predict(models: list, dataset, device: torch.device,
                     confidence_thresh: float = 0.60, batch_size: int = 512):
    all_probs = []
    labels    = None

    for m in models:
        probs, labs = predict(m, dataset, device, batch_size)
        all_probs.append(probs)
        # 2 noisy passes per model for test-time augmentation
        for _ in range(2):
            probs_noisy, _ = predict(m, dataset, device, batch_size, noise=0.01)
            all_probs.append(probs_noisy)
        if labels is None:
            labels = labs

    if not all_probs or labels is None:
        return np.array([]), np.array([]), np.array([])

    mean_probs = np.mean(all_probs, axis=0)
    conf_mask  = (mean_probs > confidence_thresh) | (mean_probs < (1 - confidence_thresh))
    return mean_probs, labels, conf_mask
