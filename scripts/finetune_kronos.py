#!/usr/bin/env python3
"""
Kronos Fine-tuning on VN30 Data
================================
Fine-tune Kronos-small (tokenizer + predictor) trên VN30 historical data.

Usage:
  # Test 1 epoch, 1 mã:
  python scripts/finetune_kronos.py --tickers VCB --epochs 1

  # Full 30 mã, 3 epoch:
  python scripts/finetune_kronos.py --epochs 3

  # Chỉ fine-tune predictor (dùng tokenizer gốc):
  python scripts/finetune_kronos.py --skip-tokenizer --epochs 3

  # LoRA-style (freeze encoder, chỉ train top 2 layers):
  python scripts/finetune_kronos.py --lora-style

Output:
  models/finetuned/<exp_name>/tokenizer/best_model/
  models/finetuned/<exp_name>/basemodel/best_model/
  models/finetuned/<exp_name>/basemodel/epoch_N/
  models/finetuned/<exp_name>/logs/
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ── Project paths ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
KRONOS_DIR = BASE_DIR / "models" / "Kronos-repo"
DATA_DIR   = BASE_DIR / "data" / "processed"
FT_OUT     = BASE_DIR / "models" / "finetuned"

sys.path.insert(0, str(KRONOS_DIR))
sys.path.insert(0, str(KRONOS_DIR / "finetune_csv"))

from model import Kronos, KronosTokenizer  # noqa: E402

# ── HuggingFace cached model paths ───────────────────────────────────────────
PRED_PATH = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--NeoQuasar--Kronos-small"
    / "snapshots"
    / "901c26c1332695a2a8f243eb2f37243a37bea320"
)
TOK_PATH = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--NeoQuasar--Kronos-Tokenizer-base"
    / "snapshots"
    / "0e0117387f39004a9016484a186a908917e22426"
)

# All 30 VN30 tickers
ALL_TICKERS = sorted(f.stem for f in DATA_DIR.glob("*.csv") if f.stem != "_summary")

# ── Dataset ───────────────────────────────────────────────────────────────────
FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]
TRAIN_END    = "2025-10-31"
VAL_START    = "2025-11-01"


class VN30Dataset(Dataset):
    """
    Sliding-window dataset cho Kronos fine-tuning từ VN30 CSV.

    - Train split: 2021-11-02 → 2025-10-31  (~80%)
    - Val   split: 2025-11-01 → 2026-10-30  (~20%)

    Val windows: dùng toàn bộ data nhưng chỉ lấy windows mà phần lớn
    nằm trong khoảng val (tránh data leak). Lookback tự động thu hẹp
    nếu val period quá ngắn so với window.
    """

    def __init__(
        self,
        csv_paths: list[Path],
        split: str = "train",       # 'train' | 'val'
        lookback: int = 400,
        pred_len: int = 20,
        stride: int = 5,
        clip: float = 5.0,
        seed: int = 42,
    ):
        self.split   = split
        self.clip    = clip
        self.rng     = random.Random(seed)
        self.segments: list[tuple[np.ndarray, np.ndarray]] = []

        # Pre-compute a SINGLE eff_lookback for val (same across all tickers)
        # to ensure uniform tensor sizes for DataLoader collation.
        if split == "val":
            all_val_rows = []
            for p in csv_paths:
                try:
                    _df = pd.read_csv(p, parse_dates=["date"])
                    _n = len(_df)
                    _dates = pd.to_datetime(_df["date"])
                    _train_end = int((_dates <= TRAIN_END).sum())
                    all_val_rows.append(_n - _train_end)
                except Exception:
                    pass
            min_val_rows = min(all_val_rows) if all_val_rows else pred_len * 3
            _fixed_eff_lookback = min(lookback, max(pred_len * 3, min_val_rows // 2))
        else:
            _fixed_eff_lookback = lookback  # unused for train

        for path in csv_paths:
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()

            # Timestamp features — must match TemporalEmbedding(module.py):
            # position 0: minute (0-59)   position 1: hour (0-23)
            # position 2: weekday (0-6)   position 3: day (1-31)
            # position 4: month (1-12)
            # Daily VN30 data: minute=0, hour=9 (market open)
            df["te_minute"]  = np.int64(0)
            df["te_hour"]    = np.int64(9)
            df["te_weekday"] = df["date"].dt.weekday.astype(np.int64)
            df["te_day"]     = df["date"].dt.day.astype(np.int64)
            df["te_month"]   = df["date"].dt.month.astype(np.int64)

            feat  = df[FEATURE_COLS].values.astype(np.float32)
            stamp = df[["te_minute", "te_hour", "te_weekday", "te_day", "te_month"]].values.astype(np.int64)
            n     = len(df)
            dates = pd.to_datetime(df["date"])

            train_end_pos = int((dates <= TRAIN_END).sum())  # index after last train row

            if split == "train":
                # Effective lookback for train: full lookback
                window      = lookback + pred_len + 1
                start_range = range(0, train_end_pos - window + 1, stride)
            else:
                # Val: ALL tickers use the SAME eff_lookback (pre-computed above)
                # so window size is identical across tickers → no collation error
                eff_lookback       = _fixed_eff_lookback
                window             = eff_lookback + pred_len + 1
                first_start        = max(0, train_end_pos - eff_lookback)
                eff_stride         = max(1, stride // 2)
                start_range        = range(first_start, n - window + 1, eff_stride)

            if n < window:
                continue

            for start in start_range:
                self.segments.append((feat[start:start + window],
                                      stamp[start:start + window]))

        if not self.segments:
            raise ValueError(
                f"No valid segments for split='{split}'. "
                "Val period may be too short for the window size. "
                "Try --lookback 100 or reduce --pred-len."
            )

        print(f"[{split.upper()}] {len(csv_paths)} ticker(s), "
              f"{len(self.segments)} windows (window≈{window})")

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, stamp = self.segments[idx]

        # Per-window z-score normalisation (same as finetune_base_model.py)
        x_mean = x.mean(axis=0)
        x_std  = x.std(axis=0)
        x_norm = (x - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -self.clip, self.clip)

        # x_norm: float32, stamp: int64 (TemporalEmbedding requires .long())
        return torch.from_numpy(x_norm), torch.from_numpy(stamp)  # stamp is int64


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(exp_name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"kronos_ft_{exp_name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler(
        log_dir / f"finetune_{exp_name}.log",
        maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Device: MPS (Apple Silicon)")
        return torch.device("mps")
    print("Device: CPU")
    return torch.device("cpu")


# ── Load pretrained models ────────────────────────────────────────────────────
def _load_state(path: Path) -> dict:
    sf = path / "model.safetensors"
    pt = path / "pytorch_model.bin"
    if sf.exists():
        from safetensors.torch import load_file
        return load_file(str(sf))
    elif pt.exists():
        return torch.load(str(pt), map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"No model weights found in {path}")


def load_tokenizer(path: Path) -> KronosTokenizer:
    with open(path / "config.json") as f:
        c = json.load(f)
    tok = KronosTokenizer(
        d_in=c["d_in"], d_model=c["d_model"], n_heads=c["n_heads"],
        ff_dim=c["ff_dim"], n_enc_layers=c["n_enc_layers"], n_dec_layers=c["n_dec_layers"],
        ffn_dropout_p=c["ffn_dropout_p"], attn_dropout_p=c["attn_dropout_p"],
        resid_dropout_p=c["resid_dropout_p"], s1_bits=c["s1_bits"], s2_bits=c["s2_bits"],
        beta=c["beta"], gamma0=c["gamma0"], gamma=c["gamma"],
        zeta=c["zeta"], group_size=c["group_size"],
    )
    tok.load_state_dict(_load_state(path), strict=False)
    return tok


def load_predictor(path: Path) -> Kronos:
    with open(path / "config.json") as f:
        c = json.load(f)
    model = Kronos(
        s1_bits=c["s1_bits"], s2_bits=c["s2_bits"], n_layers=c["n_layers"],
        d_model=c["d_model"], n_heads=c["n_heads"], ff_dim=c["ff_dim"],
        ffn_dropout_p=c["ffn_dropout_p"], attn_dropout_p=c["attn_dropout_p"],
        resid_dropout_p=c["resid_dropout_p"], token_dropout_p=c["token_dropout_p"],
        learn_te=c["learn_te"],
    )
    model.load_state_dict(_load_state(path), strict=False)
    return model


# ── LoRA-style freeze ──────────────────────────────────────────────────────────
def apply_lora_style_freeze(model: Kronos, unfreeze_last_n: int = 2):
    """Freeze all params except last N transformer layers + head."""
    for p in model.parameters():
        p.requires_grad = False

    # Find transformer layers list
    layers = None
    for attr in ("transformer", "layers", "blocks"):
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, "__iter__"):
            layers = list(candidate)
            break

    if layers:
        for layer in layers[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True

    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"LoRA-style: {trainable:,}/{total:,} params trainable ({100*trainable/total:.1f}%)")


# ── Phase 1: Tokenizer fine-tuning ───────────────────────────────────────────
def train_tokenizer_phase(
    tokenizer:    KronosTokenizer,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    save_dir:     Path,
    epochs:       int,
    lr:           float,
    logger:       logging.Logger,
    patience:     int = 2,
) -> float:
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.05, div_factor=10,
    )
    best_val, no_improve = float("inf"), 0
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        t0 = time.time()
        tokenizer.train()
        epoch_loss, nb = 0.0, 0

        for step, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            (recon_pre, recon), bsq_loss, _, _ = tokenizer(batch_x)
            rec_loss = F.mse_loss(recon, batch_x) + 0.5 * F.mse_loss(recon_pre, batch_x)
            loss = rec_loss + bsq_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 3.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item(); nb += 1
            if (step + 1) % 50 == 0:
                logger.info(f"[Tok E{epoch+1}/{epochs} S{step+1}/{len(train_loader)}] "
                            f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        tokenizer.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                (rp, r), bsq_l, _, _ = tokenizer(batch_x)
                rl = F.mse_loss(r, batch_x) + 0.5 * F.mse_loss(rp, batch_x)
                val_loss += (rl + bsq_l).item(); vn += 1

        avg_t = epoch_loss / nb if nb else 0
        avg_v = val_loss / vn if vn else float("inf")
        msg = (f"[Tok] Epoch {epoch+1}/{epochs}  train={avg_t:.4f}  "
               f"val={avg_v:.4f}  time={time.time()-t0:.1f}s")
        logger.info(msg); print(msg)

        if avg_v < best_val:
            best_val = avg_v; no_improve = 0
            best_path = save_dir / "best_model"
            best_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(best_path))
            logger.info(f"  ✓ Tokenizer saved → {best_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  Early stopping (tokenizer)."); break

    return best_val


# ── Phase 2: Predictor fine-tuning ───────────────────────────────────────────
def train_predictor_phase(
    model:        Kronos,
    tokenizer:    KronosTokenizer,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    save_dir:     Path,
    epochs:       int,
    lr:           float,
    logger:       logging.Logger,
    patience:     int = 1,
) -> tuple[float, Path]:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.03, div_factor=10,
    )
    best_val, no_improve = float("inf"), 0
    best_path = save_dir / "best_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        epoch_loss, nb = 0.0, 0

        for step, (batch_x, batch_stamp) in enumerate(train_loader):
            batch_x     = batch_x.to(device)
            batch_stamp = batch_stamp.to(device)

            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_in  = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:],  token_seq_1[:, 1:]]

            logits = model(token_in[0], token_in[1], batch_stamp[:, :-1, :])
            loss, s1l, s2l = model.head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1]
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item(); nb += 1
            if (step + 1) % 50 == 0:
                logger.info(f"[Pred E{epoch+1}/{epochs} S{step+1}/{len(train_loader)}] "
                            f"loss={loss.item():.4f} s1={s1l.item():.4f} "
                            f"s2={s2l.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        # Validation
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_stamp in val_loader:
                batch_x     = batch_x.to(device)
                batch_stamp = batch_stamp.to(device)
                t0_v, t1_v = tokenizer.encode(batch_x, half=True)
                ti = [t0_v[:, :-1], t1_v[:, :-1]]
                to = [t0_v[:, 1:],  t1_v[:, 1:]]
                lg = model(ti[0], ti[1], batch_stamp[:, :-1, :])
                vl, _, _ = model.head.compute_loss(lg[0], lg[1], to[0], to[1])
                val_loss += vl.item(); vn += 1

        avg_t = epoch_loss / nb if nb else 0
        avg_v = val_loss / vn if vn else float("inf")
        elapsed = time.time() - t0

        # Save per-epoch checkpoint
        ckpt = save_dir / f"epoch_{epoch+1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt))

        msg = (f"[Pred] Epoch {epoch+1}/{epochs}  train={avg_t:.4f}  "
               f"val={avg_v:.4f}  time={elapsed:.1f}s  ckpt={ckpt.name}")
        logger.info(msg); print(msg)

        if avg_v < best_val:
            best_val = avg_v; no_improve = 0
            best_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_path))
            logger.info(f"  ✓ Best predictor → {best_path}  (val={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  Early stopping (predictor)."); break

    return best_val, best_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Kronos-small on VN30")
    parser.add_argument("--tickers",        nargs="+", default=None,
                        help="Danh sách tickers (default: tất cả 30 mã VN30)")
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--tok-epochs",     type=int,   default=None,
                        help="Override epochs cho tokenizer phase")
    parser.add_argument("--pred-epochs",    type=int,   default=None,
                        help="Override epochs cho predictor phase")
    parser.add_argument("--lr-tok",         type=float, default=2e-4)
    parser.add_argument("--lr-pred",        type=float, default=1e-4)
    parser.add_argument("--batch",          type=int,   default=4,
                        help="Batch size (default: 4 phù hợp CPU/MPS)")
    parser.add_argument("--lookback",       type=int,   default=400)
    parser.add_argument("--pred-len",       type=int,   default=20)
    parser.add_argument("--stride",         type=int,   default=5)
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-predictor", action="store_true")
    parser.add_argument("--lora-style",     action="store_true",
                        help="Freeze tất cả trừ 2 layers cuối + head")
    parser.add_argument("--exp-name",       type=str,   default=None)
    parser.add_argument("--workers",        type=int,   default=0,
                        help="DataLoader num_workers (0 cho MPS)")
    parser.add_argument("--patience",       type=int,   default=1,
                        help="Early stopping patience for predictor (default: 1)")
    parser.add_argument("--tok-patience",   type=int,   default=2,
                        help="Early stopping patience for tokenizer (default: 2)")
    args = parser.parse_args()

    # ── Resolve tickers & paths ──
    tickers = sorted({t.upper() for t in (args.tickers or ALL_TICKERS)})
    csv_paths = [DATA_DIR / f"{t}.csv" for t in tickers if (DATA_DIR / f"{t}.csv").exists()]
    missing = [t for t in tickers if not (DATA_DIR / f"{t}.csv").exists()]
    if missing:
        print(f"WARNING: CSV not found for: {missing}")
    if not csv_paths:
        print("ERROR: No CSV files found."); sys.exit(1)

    tok_epochs  = args.tok_epochs  if args.tok_epochs  is not None else args.epochs
    pred_epochs = args.pred_epochs if args.pred_epochs is not None else args.epochs

    exp_name = args.exp_name or f"vn30_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir  = FT_OUT / exp_name
    log_dir  = exp_dir / "logs"
    tok_dir  = exp_dir / "tokenizer"
    pred_dir = exp_dir / "basemodel"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(exp_name, log_dir)
    logger.info("=" * 60)
    logger.info(f"Experiment  : {exp_name}")
    logger.info(f"Tickers     : {tickers}  ({len(csv_paths)} files)")
    logger.info(f"Lookback    : {args.lookback}  PredLen: {args.pred_len}  Stride: {args.stride}")
    logger.info(f"Batch       : {args.batch}  TokE: {tok_epochs}  PredE: {pred_epochs}")
    logger.info(f"LR tok/pred : {args.lr_tok} / {args.lr_pred}  LoRA: {args.lora_style}  Patience tok/pred: {args.tok_patience}/{args.patience}")
    logger.info("=" * 60)

    device = get_device()

    # ── Build datasets ──
    print("\nBuilding datasets …")
    train_ds = VN30Dataset(
        csv_paths, split="train",
        lookback=args.lookback, pred_len=args.pred_len, stride=args.stride,
    )
    val_ds = VN30Dataset(
        csv_paths, split="val",
        lookback=args.lookback, pred_len=args.pred_len, stride=args.stride,
    )

    kw = dict(batch_size=args.batch, num_workers=args.workers,
               pin_memory=False, drop_last=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    logger.info(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1: Tokenizer
    # ──────────────────────────────────────────────────────────────────────────
    tok_best_path = tok_dir / "best_model"

    if not args.skip_tokenizer:
        print("\n" + "=" * 60)
        print("Phase 1: Fine-tuning KronosTokenizer")
        print("=" * 60)

        tokenizer = load_tokenizer(TOK_PATH).to(device)
        logger.info(f"Tokenizer params: {sum(p.numel() for p in tokenizer.parameters()):,}")

        t0 = time.time()
        bv = train_tokenizer_phase(
            tokenizer, train_loader, val_loader, device,
            tok_dir, tok_epochs, args.lr_tok, logger,
            patience=args.tok_patience,
        )
        elapsed = time.time() - t0
        logger.info(f"Tokenizer done. best_val={bv:.4f}  time={elapsed/60:.1f}min")
        print(f"\n✓ Tokenizer fine-tuned — {elapsed/60:.1f} min  val={bv:.4f}")
        del tokenizer
    else:
        print("\nSkipping tokenizer phase.")
        if not tok_best_path.exists():
            logger.info("No finetuned tokenizer found — using original pretrained.")
            tok_best_path = TOK_PATH

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2: Predictor
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_predictor:
        print("\n" + "=" * 60)
        print("Phase 2: Fine-tuning Kronos Predictor")
        print("=" * 60)

        # Use finetuned tokenizer (frozen) for encoding
        tokenizer = load_tokenizer(tok_best_path).to(device).eval()
        for p in tokenizer.parameters():
            p.requires_grad = False

        model = load_predictor(PRED_PATH)
        if args.lora_style:
            apply_lora_style_freeze(model, unfreeze_last_n=2)
        model = model.to(device)

        n_total   = sum(p.numel() for p in model.parameters())
        n_train   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Predictor params: {n_total:,}  trainable: {n_train:,}")

        t0 = time.time()
        bv, best_ckpt = train_predictor_phase(
            model, tokenizer, train_loader, val_loader, device,
            pred_dir, pred_epochs, args.lr_pred, logger,
            patience=args.patience,
        )
        elapsed = time.time() - t0

        logger.info(f"Predictor done. best_val={bv:.4f}  time={elapsed/60:.1f}min")
        print(f"\n✓ Predictor fine-tuned — {elapsed/60:.1f} min  val={bv:.4f}")
        print(f"  Best checkpoint → {best_ckpt}")

        # Estimated full run time
        time_per_ticker = elapsed / len(csv_paths)
        est_full_min    = time_per_ticker * 30 * pred_epochs / 60
        print(f"\n  Estimated full (30 mã × {pred_epochs} epoch(s)): "
              f"~{est_full_min:.0f} min  ({est_full_min/60:.1f} hrs)")
    else:
        print("\nSkipping predictor phase.")

    print(f"\n✓ Done!  outputs → {exp_dir}")


if __name__ == "__main__":
    main()
