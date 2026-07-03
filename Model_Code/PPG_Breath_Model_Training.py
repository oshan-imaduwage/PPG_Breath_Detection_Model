from __future__ import annotations
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append("/content/drive/MyDrive/Colab_Notebooks")
from PPG_Breath_Model_Preprocessing import Config
from PPG_Breath_Model_Architecture import RespiratoryDetector

# ─────────────────────────────────────────────────────────────────────────────
# MASKED COMPOSITE LOSS
# ─────────────────────────────────────────────────────────────────────────────

class MaskedCompositeLoss(nn.Module):
    """
    L = α · L_cls + (1 - α) · L_reg

    L_cls
    -----
    BCEWithLogitsLoss applied to logits from BOTH the fusion and primary-only
    paths. Positive-class weight β > 1 penalises missed breath boundaries
    more heavily than false alarms (asymmetric clinical cost).

    L_reg
    -----
    MSELoss with a BOOLEAN MASK that zeroes out gradients for windows where
    no transition is present (t_loc == -1).  This is the critical design
    decision from Module 5: without the mask, the model is trained to predict
    a location even when no transition exists, which corrupts the regressor.

    The regression loss is computed for BOTH pathways during training and
    averaged. During inference only the fusion path's loc is used.

    Parameters
    ----------
    cfg : Config
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.alpha = cfg.ALPHA
        self.n_tr  = cfg.NUM_TRANSITIONS

        # pos_weight must be a tensor on the same device as logits
        # (moved automatically by .to(device) on the parent model)
        self.register_buffer(
            "pos_weight",
            torch.full((cfg.NUM_TRANSITIONS,), cfg.POS_WEIGHT)
        )

    def forward(self, preds: dict, targets: dict) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        preds   : output dict from RespiratoryDetector.forward()
        targets : dict with
            "t_prob" : (B, 2) float  ∈ {0, 1}
            "t_loc"  : (B, 2) float  — sample index or -1.0

        Returns dict with keys: "total", "cls", "reg"
        """
        t_prob = targets["t_prob"].float()   # (B, 2)
        t_loc  = targets["t_loc"].float()    # (B, 2)

        # ── Classification loss (both pathways) ──────────────────────────────
        # BCEWithLogitsLoss handles the sigmoid internally → numerically stable
        bce_fn = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(t_prob.device)
        )
        l_cls_fusion  = bce_fn(preds["logits_fusion"],  t_prob)
        l_cls_primary = bce_fn(preds["logits_primary"], t_prob)
        l_cls = (l_cls_fusion + l_cls_primary) / 2.0

        # ── Regression loss with masking ──────────────────────────────────────
        # mask shape: (B, 2) — True where a transition actually exists
        mask = (t_loc >= 0.0).float()                  # (B, 2)

        # Clamp t_loc so that -1 sentinels don't contribute large squared errors
        loc_gt_safe = t_loc.clamp(min=0.0)

        # Fusion path regression
        sq_err_fusion = (preds["loc"] - loc_gt_safe) ** 2   # (B, 2)
        n_valid = mask.sum().clamp(min=1.0)
        l_reg_fusion = (sq_err_fusion * mask).sum() / n_valid

        # Primary-only path regression (training only)
        if preds["loc_primary"] is not None:
            sq_err_primary = (preds["loc_primary"] - loc_gt_safe) ** 2
            l_reg_primary  = (sq_err_primary * mask).sum() / n_valid
            l_reg = (l_reg_fusion + l_reg_primary) / 2.0
        else:
            l_reg = l_reg_fusion

        total = self.alpha * l_cls + (1.0 - self.alpha) * l_reg
        return {"total": total, "cls": l_cls, "reg": l_reg}

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION LOOPS
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      RespiratoryDetector,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    loss_fn:    MaskedCompositeLoss,
    scaler:     torch.cuda.amp.GradScaler,
    cfg:        Config,
    device:     torch.device,
    epoch:      int,
) -> dict[str, float]:
    """
    One training epoch with gradient accumulation (GRAD_ACCUM steps).

    AMP (automatic mixed precision) is enabled via GradScaler.
    Gradients are clipped to max_norm=5.0 before each optimiser step
    to guard against exploding gradients from the masked MSE loss.

    Returns dict of mean losses over the epoch.
    """
    model.train()
    totals     = defaultdict(float)
    n_batches  = len(loader)
    accum      = cfg.GRAD_ACCUM

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        primary     = batch["primary"].to(device, non_blocking=True)
        auxiliary   = batch["auxiliary"].to(device, non_blocking=True)
        aux_marker1 = batch["aux_marker1"].to(device, non_blocking=True)
        t_prob      = batch["t_prob"].to(device, non_blocking=True)
        t_loc       = batch["t_loc"].to(device, non_blocking=True)

        # ── Forward pass (AMP) ────────────────────────────────────────────
        with torch.cuda.amp.autocast():
            preds  = model(primary, auxiliary, aux_marker1)
            losses = loss_fn(preds, {"t_prob": t_prob, "t_loc": t_loc})
            # Scale loss by accumulation steps so the effective loss magnitude
            # is the same regardless of GRAD_ACCUM
            scaled_loss = losses["total"] / accum

        # ── Backward ─────────────────────────────────────────────────────
        scaler.scale(scaled_loss).backward()

        # ── Optimiser step (every GRAD_ACCUM mini-batches) ───────────────
        if (step + 1) % accum == 0 or (step + 1) == n_batches:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for k, v in losses.items():
            totals[k] += v.item()

    return {k: v / n_batches for k, v in totals.items()}

@torch.no_grad()
def evaluate(
    model:    RespiratoryDetector,
    loader:   DataLoader,
    loss_fn:  MaskedCompositeLoss,
    device:   torch.device,
) -> dict[str, float]:
    """Compute validation losses (no gradient accumulation, no AMP scaling)."""
    model.eval()
    totals = defaultdict(float)

    for batch in loader:
        primary     = batch["primary"].to(device, non_blocking=True)
        auxiliary   = batch["auxiliary"].to(device, non_blocking=True)
        aux_marker1 = batch["aux_marker1"].to(device, non_blocking=True)
        t_prob      = batch["t_prob"].to(device, non_blocking=True)
        t_loc       = batch["t_loc"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            preds  = model(primary, auxiliary, aux_marker1)
            losses = loss_fn(preds, {"t_prob": t_prob, "t_loc": t_loc})

        for k, v in losses.items():
            totals[k] += v.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}

# ─────────────────────────────────────────────────────────────────────────────
# FULL TRAINING DRIVER
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:      RespiratoryDetector,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:        Config,
    device:     torch.device,
) -> RespiratoryDetector:
    """
    Complete training loop with:
        ✓ AMP (mixed precision)
        ✓ Gradient accumulation (effective batch = BATCH_SIZE × GRAD_ACCUM)
        ✓ AdamW + cosine annealing LR schedule
        ✓ Early stopping on validation total loss
        ✓ Best checkpoint saving

    Kaggle-specific: uses torch.cuda.amp for the T4 GPU (supports Tensor Cores
    but not bfloat16); if using A100 replace GradScaler with bfloat16 autocast.
    """
    model  = model.to(device)
    loss_fn = MaskedCompositeLoss(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.LR,
        betas        = (0.9, 0.99),
        weight_decay = cfg.WEIGHT_DECAY,
    )

    # Cosine annealing: decay LR from cfg.LR to cfg.LR/10 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR / 10
    )

    # GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    best_val  = float("inf")
    patience  = 0
    ckpt_path = Path(cfg.CKPT_PATH)

    print(f"\n{'─'*60}")
    print(f"  Training on {device}  |  "
          f"effective batch = {cfg.BATCH_SIZE}×{cfg.GRAD_ACCUM}={cfg.BATCH_SIZE*cfg.GRAD_ACCUM}")
    print(f"  Epochs: {cfg.EPOCHS}  |  Patience: {cfg.PATIENCE}  |  "
          f"α={cfg.ALPHA}  pos_weight={cfg.POS_WEIGHT}")
    print(f"{'─'*60}\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()

        tr = train_one_epoch(model, train_loader, optimizer, loss_fn,
                             scaler, cfg, device, epoch)
        va = evaluate(model, val_loader, loss_fn, device)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d}/{cfg.EPOCHS}  "
            f"train {tr['total']:.4f} (cls {tr['cls']:.4f} reg {tr['reg']:.4f})  "
            f"val {va['total']:.4f} (cls {va['cls']:.4f} reg {va['reg']:.4f})  "
            f"lr {lr_now:.2e}  {elapsed:.1f}s"
        )

        if va["total"] < best_val:
            best_val = va["total"]
            patience = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optim_state":  optimizer.state_dict(),
                "val_loss":     best_val,
                "cfg":          cfg.__dict__,
            }, ckpt_path)
            print(f"  ✓ Saved checkpoint (val {best_val:.4f})")
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                print(f"\n  Early stopping after {epoch} epochs "
                      f"(no improvement for {cfg.PATIENCE} epochs).")
                break

    # Restore best weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n  Best val loss: {ckpt['val_loss']:.4f}  "
          f"(epoch {ckpt['epoch']})")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# KAGGLE DUAL-GPU WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def wrap_for_kaggle(model: RespiratoryDetector) -> nn.Module:
    """
    Kaggle T4×2 environments expose two GPUs via DataParallel.
    This wraps the model IF two GPUs are available.

    Note: DataParallel splits the BATCH dimension — gradient accumulation
    still operates correctly at the optimizer level.
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        print(f"  Using DataParallel across {n_gpu} GPUs.")
        return nn.DataParallel(model)
    return model

# if __name__ == "__main__":
#     # Smoke test: one forward + backward on CPU
#     import numpy as np
#     from PPG_Breath_Model_Dataset import RespiratoryWindowDataset, make_loader
#     from PPG_Breath_Model_Preprocessing import SignalProcessor

#     cfg  = Config()

#     # ── CPU testing overrides ──────────────────────────────
#     cfg.BATCH_SIZE = 16      # must be < number of windows (111 here)
#     cfg.EPOCHS     = 2
#     cfg.PATIENCE   = 2
#     cfg.CKPT_PATH  = "/content/test_checkpoint.pt"
#     # ───────────────────────────────────────────────────────

#     proc = SignalProcessor(cfg)
#     T    = 6000
#     t_ax = np.linspace(0, 60, T)
#     resp = np.sin(2 * np.pi * 0.25 * t_ax)
#     labels = proc.extract_transitions(resp)
#     fake_rec = {
#         "primary":    np.random.randn(T).astype(np.float32),
#         "auxiliary":  np.random.randn(T).astype(np.float32),
#         "aux_marker1":np.random.rand(T).astype(np.float32),
#         "labels":     labels,
#     }
#     ds     = RespiratoryWindowDataset([fake_rec], cfg)
#     loader = make_loader(ds, cfg, train=True, num_workers=0)

#     device  = torch.device("cpu")
#     model   = RespiratoryDetector(cfg).to(device)
#     loss_fn = MaskedCompositeLoss(cfg).to(device)
#     opt     = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
#     # scaler  = torch.cuda.amp.GradScaler(enabled=False)  # disabled on CPU
#     scaler = torch.amp.GradScaler('cuda',enabled = False)

#     batch  = next(iter(loader))
#     preds  = model(batch["primary"], batch["auxiliary"], batch["aux_marker1"])
#     losses = loss_fn(preds, {"t_prob": batch["t_prob"], "t_loc": batch["t_loc"]})
#     losses["total"].backward()

#     print(f"Loss: {losses['total'].item():.4f}  "
#           f"cls={losses['cls'].item():.4f}  reg={losses['reg'].item():.4f}")
#     print("Backward pass OK — gradients flow correctly.")

# import torch
# import numpy as np
# from PPG_Breath_Model_Preprocessing import Config, SignalProcessor
# from PPG_Breath_Model_Dataset import RespiratoryWindowDataset, make_loader
# from PPG_Breath_Model_Architecture import RespiratoryDetector
# # from module5_train import MaskedCompositeLoss

# cfg    = Config()
# # ── CPU testing overrides ──────────────────────────────
# cfg.BATCH_SIZE = 16      # must be < number of windows (111 here)
# cfg.EPOCHS     = 2
# cfg.PATIENCE   = 2
# cfg.CKPT_PATH  = "/content/test_checkpoint.pt"
# # ───────────────────────────────────────────────────────


# proc   = SignalProcessor(cfg)
# device = torch.device("cpu")

# T    = 6000
# t    = np.linspace(0, 60, T)
# resp = np.sin(2 * np.pi * 0.25 * t)

# fake_rec = {
#     "primary":     np.random.randn(T).astype(np.float32),
#     "auxiliary":   np.random.randn(T).astype(np.float32),
#     "aux_marker1": np.random.rand(T).astype(np.float32),
#     "labels":      proc.extract_transitions(resp),
# }

# ds      = RespiratoryWindowDataset([fake_rec], cfg)
# loader  = make_loader(ds, cfg, train=True, num_workers=0)
# model   = RespiratoryDetector(cfg).to(device)
# loss_fn = MaskedCompositeLoss(cfg).to(device)

# for batch in loader:
#     preds  = model(batch["primary"], batch["auxiliary"], batch["aux_marker1"])
#     losses = loss_fn(preds, {"t_prob": batch["t_prob"], "t_loc": batch["t_loc"]})
#     assert not torch.isnan(losses["total"]), "NaN loss detected!"
#     assert not torch.isinf(losses["total"]), "Inf loss detected!"

# print("No NaN or Inf losses — loss function is stable")

# model.zero_grad()
# batch  = next(iter(loader))
# preds  = model(batch["primary"], batch["auxiliary"], batch["aux_marker1"])
# losses = loss_fn(preds, {"t_prob": batch["t_prob"], "t_loc": batch["t_loc"]})
# losses["total"].backward()

# no_grad = []
# for name, param in model.named_parameters():
#     if param.requires_grad and param.grad is None:
#         no_grad.append(name)

# if no_grad:
#     print(f"WARNING — no gradient for: {no_grad}")
# else:
#     print("All parameters received gradients")

# from PPG_Breath_Model_Preprocessing import subject_split

# # Override paths for Colab
# cfg.CKPT_PATH = "/content/test_checkpoint.pt"
# cfg.EPOCHS    = 2
# cfg.PATIENCE  = 2
# cfg.BATCH_SIZE = 8   # small for CPU speed

# train_recs, val_recs = subject_split([fake_rec, fake_rec, fake_rec], val_fraction=0.33)

# train_ds = RespiratoryWindowDataset(train_recs, cfg, augment=True)
# val_ds   = RespiratoryWindowDataset(val_recs,   cfg, augment=False)

# train_loader = make_loader(train_ds, cfg, train=True,  num_workers=0)
# val_loader   = make_loader(val_ds,   cfg, train=False, num_workers=0)

# model  = RespiratoryDetector(cfg)
# model  = train(model, train_loader, val_loader, cfg, device)
# print("Full training loop ran without errors")

# import torch
# import random
# from pathlib import Path

# # --- Imports from your custom modules ---
# # Ensure these match your actual file/class names!
# from PPG_Breath_Model_Preprocessing import Config, BIDMCLoader
# from PPG_Breath_Model_Dataset import RespiratoryWindowDataset, make_loader
# from PPG_Breath_Model_Architecture import RespiratoryDetector

# # ==========================================
# # 1. CONFIGURATION FOR A QUICK TEST
# # ==========================================
# cfg = Config()
# cfg.EPOCHS = 2          # Override to only run 2 epochs!
# # cfg.BATCH_SIZE = 128  # Uncomment and lower this if your GPU runs out of memory (OOM error)

# print(f"Starting pipeline on {cfg.EPOCHS} epochs...")

# # ==========================================
# # 2. THE FARMER: LOAD ALL PATIENTS
# # ==========================================
# print("\nLoading raw BIDMC dataset from hard drive...")
# loader = BIDMCLoader(cfg)
# all_recordings = loader.load_all()

# # ==========================================

# # ==========================================
# # 3. THE SPLIT: TRAIN VS VALIDATION
# # ==========================================
# # Shuffle the patients randomly so we don't always train on the exact same ones
# random.seed(42) # Set seed for reproducibility
# random.shuffle(all_recordings)

# # 80% for training, 20% for testing (validation)
# split_idx = int(len(all_recordings) * 0.8)
# train_recs = all_recordings[:split_idx]
# val_recs   = all_recordings[split_idx:]

# print(f"🔀 Split data: {len(train_recs)} patients for Training, {len(val_recs)} for Validation.")

# # ==========================================
# # 4. THE CHEF: CREATE DATASETS & LOADERS
# # ==========================================
# print("\nPreparing PyTorch Datasets...")
# # Turn ON augmentation for training to make the model robust
# train_ds = RespiratoryWindowDataset(train_recs, cfg, augment=True)

# # Turn OFF augmentation for validation (we want to test on pure, unaltered real-world data)
# val_ds = RespiratoryWindowDataset(val_recs, cfg, augment=False)

# print(f"✅ Train Windows: {len(train_ds)} | Val Windows: {len(val_ds)}")

# # Create the Waiters (Loaders)
# # num_workers=0 is usually safest for Kaggle/Colab quick tests
# train_loader = make_loader(train_ds, cfg, train=True, num_workers=0)
# val_loader   = make_loader(val_ds, cfg, train=False, num_workers=0)

# # ==========================================
# # 5. THE BRAIN: INITIALIZE MODEL
# # ==========================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"\nInitializing Neural Network on {device}...")
# model = RespiratoryDetector(cfg)

# # ==========================================
# # 6. TEACH: THE TRAINING LOOP
# # ==========================================
# print("\n🔥 STARTING TRAINING LOOP 🔥")
# # This calls the train() function from module5_train.py!
# trained_model = train(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     cfg=cfg,
#     device=device
# )

# print("\n🎉 TRAINING COMPLETE! The best model weights have been saved.")