"""Track A inference (time domain ONLY): complex I/Q -> RMS normalisation -> lag conjugate products -> 1D CNN/GRU.
No frequency-domain / time-frequency transform is used anywhere in this file.

Self-test : python src/track_a/predict_track_a.py --selftest
Predict   : python src/track_a/predict_track_a.py            (writes submissions/predictions_track_a.csv)
Needs: numpy, torch.   Weights: track_a_runs/final/model.pt + config.json
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

WEIGHTS_DIR = ROOT / "track_a_runs" / "final"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = json.load(open(WEIGHTS_DIR / "config.json"))
LAGS = tuple(CFG["lags"])
S0, S1 = CFG["crop_samples"]
assert CFG["classes"] == CLASSES


def features_from_z(z):
    """z: (B, L) complex -> (B, 2 + 2*len(LAGS), L).  Identical to training (without the random phase)."""
    z = z / (z.abs().pow(2).mean(1, keepdim=True).sqrt() + 1e-6)
    ch = [z.real, z.imag]
    for k in LAGS:
        p = z[:, k:] * torch.conj(z[:, :-k])
        p = F.pad(p, (k, 0))
        ch += [p.real, p.imag]
    return torch.stack(ch, dim=1)


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(c, max(c // r, 8)), nn.Linear(max(c // r, 8), c)
    def forward(self, x):
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(x.mean(2)))))
        return x * w.unsqueeze(2)


class Block(nn.Module):
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False); self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False);     self.b2 = nn.BatchNorm1d(cout)
        self.se = SE(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.AvgPool1d(stride, stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                                       nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout))
    def forward(self, x):
        h = F.silu(self.b1(self.c1(x)))
        h = self.se(self.b2(self.c2(h)))
        s = x if self.short is None else self.short(x)
        if s.shape[-1] != h.shape[-1]:
            s = s[..., :h.shape[-1]]
        return F.silu(h + s)


class AttnStats(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(c, c // 2), nn.Tanh(), nn.Linear(c // 2, 1))
    def forward(self, h):
        w = torch.softmax(self.att(h), dim=1)
        mu = (w * h).sum(1)
        sd = torch.sqrt((w * (h - mu.unsqueeze(1)) ** 2).sum(1) + 1e-5)
        return torch.cat([mu, sd], dim=1)


class TrackANet(nn.Module):
    def __init__(self, n_in, widths, blocks, kernels, gru=128, drop=0.2, n_cls=7):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_in, 32, 31, 2, 15, bias=False), nn.BatchNorm1d(32), nn.SiLU())
        layers, c = [], 32
        for w, nb, k in zip(widths, blocks, kernels):
            for j in range(nb):
                layers.append(Block(c, w, k, 2 if j == 0 else 1)); c = w
        self.body = nn.Sequential(*layers)
        self.gru = nn.GRU(c, gru, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(2 * gru)
        self.pool = AttnStats(2 * gru)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(4 * gru, 256), nn.SiLU(), nn.Dropout(drop), nn.Linear(256, n_cls))
    def forward(self, x):
        h = self.body(self.stem(x)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(self.pool(self.ln(h)))


net = TrackANet(CFG["n_in"], tuple(CFG["widths"]), tuple(CFG["blocks"]), tuple(CFG["kernels"]), CFG["gru"]).to(device)
net.load_state_dict(torch.load(WEIGHTS_DIR / "model.pt", map_location=device, weights_only=True))
net.eval()


@torch.no_grad()
def predict_fn(z_np):
    """z_np: (B, 16800) complex64 pulses (as returned by participant_loader) -> (B,) class indices."""
    z = torch.from_numpy(z_np).to(device)[:, S0:S1]
    return net(features_from_z(z)).argmax(1).cpu().numpy()


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
    main_cli("A", ROOT / "submissions" / "predictions_track_a.csv", predict_fn,
             dict(weights_sha256=sha256(WEIGHTS_DIR / "model.pt"), device=str(device), lags=list(LAGS), crop=[S0, S1]))
