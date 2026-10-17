"""Track B inference (time-frequency): STFT log-spectrogram (noise-floor whitening, signal-window crop) -> 3 residual 2D CNNs,
probabilities averaged.  The STFT is part of the timed inference.

Self-test : python src/track_b/predict_track_b.py --selftest
Predict   : python src/track_b/predict_track_b.py           (writes submissions/predictions_track_b.csv)
Needs: numpy, torch.   Weights: track_b_plus/best_nfft*_hop*_seed*.pt + ensemble_config.json
"""
import argparse, csv, hashlib, json, os, sys, time
from collections import Counter
from pathlib import Path
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # Radar_ML_lite folder
DATA_DIR = ROOT / "dataset"
SPLIT_CSV = ROOT / "dataset_infos" / "split_manifest.csv"

FS = 120e6
CLASSES = ["barker", "cw", "fmcw", "fsk", "hyp", "nlfm", "quad"]
N0, S0, S1 = int(20e-6 * FS), int(21e-6 * FS), int(119e-6 * FS)

WEIGHTS_DIR = ROOT / "track_b_plus"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = json.load(open(WEIGHTS_DIR / "ensemble_config.json"))
CROP, WHITEN = bool(CFG["crop"]), bool(CFG["whiten"])
assert CFG["classes"] == CLASSES
_WIN = {}


def get_win(n):
    if n not in _WIN:
        _WIN[n] = torch.hann_window(n).to(device)
    return _WIN[n]


def frame_ranges(nfft, hop):
    n_noise = max(1, (N0 - nfft) // hop + 1)
    k0 = -(-S0 // hop)
    k1 = (S1 - nfft) // hop + 1
    return n_noise, k0, k1


def to_spec(x, nfft, hop):
    """x: (B, 33600) interleaved I/Q float -> (B, 1, nfft, frames) normalised log-spectrogram (same as training)."""
    iq = torch.complex(x[:, 0::2], x[:, 1::2])
    S = torch.stft(iq, n_fft=nfft, hop_length=hop, window=get_win(nfft), center=False,
                   return_complex=True, onesided=False)
    P = torch.fft.fftshift(S.abs() ** 2, dim=1)
    n_noise, k0, k1 = frame_ranges(nfft, hop)
    L = 10 * torch.log10(P + 1e-8)
    if WHITEN:
        floor = P[:, :, :n_noise].mean(2, keepdim=True)
        L = L - 10 * torch.log10(floor + 1e-8)
    if CROP:
        L = L[:, :, k0:k1]
    L = (L - L.mean((1, 2), keepdim=True)) / (L.std((1, 2), keepdim=True) + 1e-6)
    return L.unsqueeze(1)


class ResBlock(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, stride=stride, padding=1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, padding=1, bias=False);                self.b2 = nn.BatchNorm2d(o)
        if i == o and stride == 1:
            self.short = nn.Identity()
        else:
            self.short = nn.Sequential(nn.Conv2d(i, o, 1, stride=stride, bias=False), nn.BatchNorm2d(o))
    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(y + self.short(x))


class Net(nn.Module):
    def __init__(self, n_classes=7, width=32, drop=0.3):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(1, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        self.body = nn.Sequential(ResBlock(w, 2 * w, 2), ResBlock(2 * w, 4 * w, 2), ResBlock(4 * w, 8 * w, 2), ResBlock(8 * w, 8 * w, 2))
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(8 * w * 16, n_classes))
    def forward(self, x):
        return self.head(self.pool(self.body(self.stem(x))))


MODELS = []
for nfft, hop, seed in CFG["configs"]:
    m = Net().to(device)
    m.load_state_dict(torch.load(WEIGHTS_DIR / f"best_nfft{nfft}_hop{hop}_seed{seed}.pt", map_location=device, weights_only=True))
    MODELS.append((m.eval(), int(nfft), int(hop)))
print(f"loaded {len(MODELS)} models, CROP={CROP} WHITEN={WHITEN}, device={device}")


@torch.no_grad()
def predict_fn(z_np):
    """z_np: (B, 16800) complex64 -> (B,) class indices (average of the 3 softmax outputs)."""
    x = np.empty((z_np.shape[0], 2 * z_np.shape[1]), dtype=np.float32)
    x[:, 0::2] = z_np.real; x[:, 1::2] = z_np.imag            # back to the interleaved I/Q layout used in training
    x = torch.from_numpy(x).to(device)
    total = 0
    for m, nfft, hop in MODELS:
        total = total + torch.softmax(m(to_spec(x, nfft, hop)).float(), 1)
    return total.argmax(1).cpu().numpy()


# ----------------------------------------------------------------------------------------------
# generic helpers (blind package, hashing, batching, validation of the csv)
# ----------------------------------------------------------------------------------------------
def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_blind_dir():
    """Folder that has participant_loader.py + submission_template.csv + test/index.csv."""
    for cur, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".venv", ".git", "dataset", "__pycache__", "node_modules")]
        if "submission_template.csv" in files and "participant_loader.py" in files and (Path(cur) / "test" / "index.csv").exists():
            return Path(cur)
    raise FileNotFoundError("Blind package not found under " + str(ROOT) + ". Use --blind-dir PATH")


def verify_blind(blind_dir, everything):
    """Check the blind package against its checksums.json (small files always, big .npy with --verify-all)."""
    sums = json.load(open(blind_dir / "checksums.json"))
    bad = []
    for name, expected in sums.items():
        p = blind_dir / name
        if not p.exists():
            bad.append(name + " (missing)"); continue
        if name.endswith(".npy") and not everything:
            continue
        if sha256(p) != expected:
            bad.append(name + " (hash differs)")
    if bad:
        raise SystemExit("Blind package integrity problem: " + ", ".join(bad))
    print("blind package checksums OK" + ("" if everything else " (small files; use --verify-all for the .npy shards)"))


def load_samples(blind_dir):
    spec = importlib.util.spec_from_file_location("participant_loader", blind_dir / "participant_loader.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.Samples(blind_dir / "test")


def iter_batches(samples, bs):
    ids, zs = [], []
    for k in range(len(samples)):
        iq, info = samples[k]                       # iq: complex64 (16800,), info["sample_id"]
        ids.append(info["sample_id"]); zs.append(np.asarray(iq, dtype=np.complex64))
        if len(zs) == bs:
            yield ids, np.stack(zs); ids, zs = [], []
    if zs:
        yield ids, np.stack(zs)


def write_and_check(pred, template_path, out_path):
    """Write sample_id,predicted_label in TEMPLATE ORDER and re-read the file to check it."""
    with open(template_path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        tmpl = [r[0] for r in rd if r]
    assert header == ["sample_id", "predicted_label"], header
    assert len(tmpl) == len(set(tmpl)), "duplicate ids in template"
    missing = [s for s in tmpl if s not in pred]
    extra = set(pred) - set(tmpl)
    if missing or extra:
        raise SystemExit(f"ids mismatch: {len(missing)} missing, {len(extra)} extra")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows((s, CLASSES[pred[s]]) for s in tmpl)
    # re-read & validate
    with open(out_path, newline="") as f:
        rd = csv.reader(f)
        assert next(rd) == header
        rows = [r for r in rd]
    assert len(rows) == len(tmpl) and [r[0] for r in rows] == tmpl, "order/size mismatch"
    assert all(len(r) == 2 and r[1] in CLASSES for r in rows), "empty or invalid label"
    return Counter(r[1] for r in rows)


def balanced_accuracy(pred, y):
    rec = [float((pred[y == c] == c).mean()) for c in range(7) if (y == c).any()]
    return float(np.mean(rec))


def selftest_data(stride=4, seed=0):
    """Validation pulses (official split, every `stride`-th) + their noise/signal power. Label from filename."""
    rows = [r for r in csv.DictReader(open(SPLIT_CSV, newline="")) if r["split"] == "validation"]
    by_file = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append(int(r["pulse_index"]))
    X, y, npw, spw = [], [], [], []
    for f, idxs in by_file.items():
        idxs = np.array(sorted(idxs))[::stride]
        tok = f.split("_")[1].split("#")[0].lower()
        lab = CLASSES.index("barker" if tok.startswith("barker") else tok)
        raw = np.asarray(np.load(DATA_DIR / f, mmap_mode="r")[0][idxs])
        rf = raw.astype(np.float32); p = rf[:, 0::2] ** 2 + rf[:, 1::2] ** 2
        n = p[:, :N0].mean(1); s = p[:, S0:S1].mean(1) - n
        ok = np.isfinite(s) & (s > 0)
        X.append(raw[ok]); y.append(np.full(ok.sum(), lab)); npw.append(n[ok]); spw.append(s[ok])
    return np.concatenate(X), np.concatenate(y), np.concatenate(npw), np.concatenate(spw)


def noisy_iq(raw, npw, spw, snr_db, rng):
    """Organizers-style degradation (numpy), then back to complex64 exactly like participant_loader."""
    added = spw / (10.0 ** (snr_db / 10.0)) - npw
    sigma = np.sqrt(np.clip(added, 0, None) / 2)[:, None]
    x = np.clip(np.trunc(raw.astype(np.float32) + rng.standard_normal(raw.shape).astype(np.float32) * sigma), -32768, 32767)
    return x[:, 0::2] + 1j * x[:, 1::2]


def run_selftest(predict_fn, snrs=(0, -10, -15, -20, -25), bs=256):
    print("self-test on validation pulses (every 4th), same inference path as the submission ...")
    X, y, npw, spw = selftest_data()
    rng = np.random.default_rng(123)
    res = {}
    for snr in snrs:
        z = noisy_iq(X, npw, spw, snr, rng).astype(np.complex64)
        pred = np.concatenate([predict_fn(z[i:i + bs]) for i in range(0, len(z), bs)])
        res[snr] = balanced_accuracy(pred, y)
        print(f"  SNR {snr:+4d} dB : balanced accuracy {res[snr]:.3f}  ({len(y)} pulses)")
    return res


def main_cli(name, default_out, predict_fn, extra_meta):
    ap = argparse.ArgumentParser(description=name)
    ap.add_argument("--blind-dir", default=None, help="folder with participant_loader.py, submission_template.csv, test/")
    ap.add_argument("--out", default=str(default_out))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--selftest", action="store_true", help="only check the pipeline on validation pulses")
    ap.add_argument("--verify-all", action="store_true", help="also hash the big test .npy shards")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(predict_fn, bs=args.batch)
        return
    blind = Path(args.blind_dir).resolve() if args.blind_dir else find_blind_dir()
    print("blind package:", blind)
    verify_blind(blind, args.verify_all)
    samples = load_samples(blind)
    n = len(samples)
    print(f"{n} test pulses")
    pred, t0 = {}, time.perf_counter()
    for ids, z in iter_batches(samples, args.batch):
        labels = predict_fn(z)
        for sid, lab in zip(ids, labels.tolist()):
            if sid in pred:
                raise SystemExit("duplicate sample_id in test index: " + sid)
            pred[sid] = lab
        if len(pred) % 20480 < args.batch:
            print(f"  {len(pred)}/{n}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    secs = time.perf_counter() - t0
    out = Path(args.out).resolve()
    counts = write_and_check(pred, blind / "submission_template.csv", out)
    meta = dict(track=name, output=str(out), output_sha256=sha256(out), examples=n, seconds=round(secs, 1),
                ms_per_pulse_incl_loading=round(1000 * secs / n, 4), predicted_label_counts=dict(counts),
                torch=torch.__version__, **extra_meta)
    json.dump(meta, open(out.with_name(out.stem + "_run.json"), "w"), indent=2)
    print(f"wrote {out}  ({n} rows, {1000 * secs / n:.3f} ms/pulse incl. loading)")
    print("label counts:", dict(counts))


if __name__ == "__main__":
    main_cli("B", ROOT / "submissions" / "predictions_track_b.csv", predict_fn,
             dict(weights_sha256={p.name: sha256(p) for p in sorted(WEIGHTS_DIR.glob("best_nfft*.pt"))}, device=str(device),
                  configs=CFG["configs"], crop=CROP, whiten=WHITEN))
