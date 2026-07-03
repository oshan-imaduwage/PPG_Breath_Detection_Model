from __future__ import annotations
import numpy as np
import torch
import sys
sys.path.append("/content/drive/MyDrive/Colab_Notebooks")
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PPG_Breath_Model_Preprocessing import Config, SignalProcessor, BIDMCLoader


class RespiratoryWindowDataset(Dataset):
    """
    Sliding-window dataset for respiratory phase detection.

    Each returned item is a dict with:

        primary      : FloatTensor (1, W)  — normalised, filtered PPG
        auxiliary    : FloatTensor (1, W)  — normalised, filtered ECG
        aux_marker1  : FloatTensor (1, W)  — Gaussian R-peak indicator
        t_prob       : FloatTensor (2,)    — [has_tr1, has_tr2] ∈ {0, 1}
        t_loc        : FloatTensor (2,)    — sample index of transition, or -1.0

    Window-level per-channel normalisation (98th percentile) is applied
    independently to primary and auxiliary, as specified in Module 2.
    The aux_marker1 is already in [0,1] after Gaussian smoothing — no further
    normalisation needed.

    Parameters
    ----------
    recordings : list[dict]
        Each dict must contain "primary", "auxiliary", "aux_marker1", "labels".
    cfg : Config
    augment : bool
        If True, applies lightweight augmentation (time-shift jitter, amplitude
        scale, additive Gaussian noise). Use only on training data.
    """

    # Transition label values (must match SignalProcessor.extract_transitions)
    TR1_LABEL = 1   # inspiration onset
    TR2_LABEL = 2   # expiration onset

    def __init__(self, recordings: list[dict], cfg: Config,
                 augment: bool = False):
        self.cfg       = cfg
        self.proc      = SignalProcessor(cfg)
        self.augment   = augment
        self._windows  = []          # (rec_idx, start_sample)

        W = cfg.WINDOW_SAMPLES
        S = cfg.STRIDE_SAMPLES

        for rec_idx, rec in enumerate(recordings):
            T = len(rec["primary"])
            for start in range(0, T - W + 1, S):
                self._windows.append((rec_idx, start))

        self._recordings = recordings

        # Pre-compute per-window class weights for the sampler (see make_loader)
        self._sample_weights = self._compute_sample_weights()

 # ── Core ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        rec_idx, start = self._windows[idx]
        rec = self._recordings[rec_idx]
        W   = self.cfg.WINDOW_SAMPLES

        # ── Extract raw window slices ──
        ppg    = rec["primary"][start : start + W].astype(np.float32)
        ecg    = rec["auxiliary"][start : start + W].astype(np.float32)
        rpeak  = rec["aux_marker1"][start : start + W].astype(np.float32)
        labels = rec["labels"][start : start + W]

        # ── Optional augmentation (training only) ──
        if self.augment:
            ppg, ecg, rpeak = self._augment(ppg, ecg, rpeak)

        # ── Per-window normalisation (PPG and ECG independently) ──
        ppg   = self.proc.normalize_window(ppg)
        ecg   = self.proc.normalize_window(ecg)
        # rpeak is already in [0, 1] — no normalisation needed

        # ── Build targets ──
        t_prob, t_loc = self._build_targets(labels)

        return {
            "primary":    torch.from_numpy(ppg).unsqueeze(0),        # (1, W)
            "auxiliary":  torch.from_numpy(ecg).unsqueeze(0),        # (1, W)
            "aux_marker1":torch.from_numpy(rpeak).unsqueeze(0),      # (1, W)
            "t_prob":     torch.from_numpy(t_prob),                  # (2,)
            "t_loc":      torch.from_numpy(t_loc),                   # (2,)
        }


    # ── Target construction ─────────────────────────────────────────────────

    def _build_targets(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Scan the label array for transition events and encode them.

        Returns
        -------
        t_prob : float32 (2,)  — binary presence indicator
        t_loc  : float32 (2,)  — sample position within window, or -1.0

        Design notes
        ------------
        * We look for a sample whose label equals TR1_LABEL (1) or TR2_LABEL (2).
          The first such sample in the window is taken as the transition location.
        * At normal breathing rates (12–20/min) and our window size (500 samples =
          5 s), at most one inspiration and one expiration event fit per window.
          If two of the same type occur (fast breathing), only the first is used.
        """
        t_prob = np.zeros(2, dtype=np.float32)
        t_loc  = np.full(2, -1.0, dtype=np.float32)

        for sample_idx, lbl in enumerate(labels):
            if lbl == self.TR1_LABEL and t_prob[0] == 0:
                t_prob[0] = 1.0
                t_loc[0]  = float(sample_idx)
            elif lbl == self.TR2_LABEL and t_prob[1] == 0:
                t_prob[1] = 1.0
                t_loc[1]  = float(sample_idx)

        return t_prob, t_loc

# ── Augmentation ────────────────────────────────────────────────────────

    def _augment(self, ppg: np.ndarray, ecg: np.ndarray,
                 rpeak: np.ndarray) -> tuple:
        """
        Lightweight augmentations that preserve respiratory physiology:

        1. Amplitude scale  : multiply by U(0.8, 1.2) — simulates pressure/
                              contact variation across subjects.
        2. Gaussian noise   : σ = 0.02 — simulates sensor noise.
        3. Baseline wander  : add a slow sinusoid at 0.05–0.2 Hz — mimics
                              body motion artefacts common in wearables.
        """
        W  = len(ppg)
        fs = self.cfg.TARGET_FS

        # 1. Amplitude scale (only PPG — ECG amplitude is not critical here)
        ppg = ppg * np.random.uniform(0.8, 1.2)

        # 2. Gaussian noise on PPG and ECG
        ppg   = ppg   + np.random.normal(0, 0.02, W).astype(np.float32)
        ecg   = ecg   + np.random.normal(0, 0.01, W).astype(np.float32)

        # 3. Baseline wander on PPG (the dominant artefact in wrist wearables)
        freq  = np.random.uniform(0.05, 0.2)
        phase = np.random.uniform(0, 2 * np.pi)
        t     = np.arange(W) / fs
        wander = (np.random.uniform(0.05, 0.2)
                  * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
        ppg   = ppg + wander

        return ppg, ecg, rpeak

# ── Sampler weights ─────────────────────────────────────────────────────

    def _compute_sample_weights(self) -> np.ndarray:
        """
        Compute a per-window weight for WeightedRandomSampler.
        Windows containing at least one transition are upsampled to balance
        the majority class (no-transition windows).
        """
        weights = np.ones(len(self._windows), dtype=np.float32)
        W = self.cfg.WINDOW_SAMPLES
        S = self.cfg.STRIDE_SAMPLES


        #
        n_pos = 0
        n_neg = 0
        for i, (rec_idx, start) in enumerate(self._windows):
            labels = self._recordings[rec_idx]["labels"][start : start + W]
            has_transition = (labels == 1).any() or (labels == 2).any()
            if has_transition:
                n_pos += 1
            else:
                n_neg += 1

        if n_pos == 0 or n_neg == 0:
            return weights

        w_pos = (n_pos + n_neg) / (2.0 * n_pos)
        w_neg = (n_pos + n_neg) / (2.0 * n_neg)

        for i, (rec_idx, start) in enumerate(self._windows):
            labels = self._recordings[rec_idx]["labels"][start : start + W]
            has_tr = (labels == 1).any() or (labels == 2).any()
            weights[i] = w_pos if has_tr else w_neg

        return weights


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_loader(dataset: RespiratoryWindowDataset, cfg: Config,
                train: bool = True, num_workers: int = 4) -> DataLoader:
    """
    Build a DataLoader.
    For training: uses WeightedRandomSampler to balance transition / no-transition
    windows. For validation: sequential order, no sampling tricks.
    """
    if train:
        sampler = WeightedRandomSampler(
            weights     = torch.from_numpy(dataset._sample_weights),
            num_samples = len(dataset),
            replacement = True,
        )
        return DataLoader(
            dataset,
            batch_size  = cfg.BATCH_SIZE,
            sampler     = sampler,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = True,          # keeps gradient accumulation math clean
        )
    else:
        return DataLoader(
            dataset,
            batch_size  = cfg.BATCH_SIZE * 2,   # faster eval, no sampler overhead
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
        )

# import torch

# # 1. Load one real patient recording (using the code we built earlier)
# cfg = Config()
# loader = BIDMCLoader(cfg)
# sample = loader._load_one(f"{cfg.BIDMC_DIR}/bidmc05")

# if sample is not None:
#     # 2. Pass it into the Dataset (it expects a list of recordings)
#     # We turn ON augmentation just to see it work
#     dataset = RespiratoryWindowDataset([sample], cfg, augment=True)
#     print(f"Total windows created for this patient: {len(dataset)}")

#     # 3. Pass the Dataset into the DataLoader factory
#     # num_workers=0 is safer for quick Colab tests
#     dataloader = make_loader(dataset, cfg, train=True, num_workers=0)

#     # 4. Pull exactly one batch of data from the "hat"
#     batch = next(iter(dataloader))

#     # 5. Inspect the PyTorch Tensors!
#     print("\n--- BATCH INSPECTION ---")
#     print(f"Primary (PPG) shape: {batch['primary'].shape}")
#     print(f"Auxiliary (ECG) shape: {batch['auxiliary'].shape}")
#     print(f"R-peak marker shape: {batch['aux_marker1'].shape}")
#     print(f"Target Probabilities shape: {batch['t_prob'].shape}")
#     print(f"Target Locations shape: {batch['t_loc'].shape}")

#     # Let's see how many windows in this batch actually contain a transition
#     # Because of our WeightedRandomSampler, this shouldn't be zero!
#     transitions_in_batch = batch['t_prob'].sum().item()
#     print(f"\nTotal transition events captured in this batch: {transitions_in_batch}")

# else:
#     print("Could not load the test file.")