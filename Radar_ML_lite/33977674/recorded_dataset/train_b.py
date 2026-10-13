import csv, json, time
import numpy as np
import torch
import torch.nn as nn

# ---------- settings (change these) ----------
N_TRAIN_PER_FILE = 40    # 26,400 training pulses
N_VAL_PER_FILE = 10      # 6,600 validation pulses
EPOCHS = 15
BATCH = 64
SEED = 42
# ---------------------------------------------

FS = 120e6
CLASSES = ["barker", "cw", "fmcw", "fsk", "hyp", "nlfm", "quad"]
SNRS = [30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30]
N0, S0, S1 = int(20e-6 * FS), int(21e-6 * FS), int(119e-6 * FS)  # noise / signal windows

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def get_label(filename):
    token = filename.split("_")[1].split("#")[0].lower()
    return "barker" if token.startswith("barker") else token


def load_split(rows, per_file):
    """Load a few pulses per file into RAM (int16) + their noise/signal power."""
    by_file = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append(int(r["pulse_index"]))
    X, y, npw, spw = [], [], [], []
    for f, idxs in by_file.items():
        idxs = list(rng.permutation(idxs))[:per_file]
        arr = np.load(f, mmap_mode="r")
        label = CLASSES.index(get_label(f))
        for i in idxs:
            raw = np.array(arr[0, int(i)])
            iq = raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)
            n = np.mean(np.abs(iq[:N0]) ** 2)
            s = np.mean(np.abs(iq[S0:S1]) ** 2) - n
            if not np.isfinite(s) or s <= 0:
                continue
            X.append(raw); y.append(label); npw.append(n); spw.append(s)
    return (np.stack(X), np.array(y), np.array(npw, np.float32), np.array(spw, np.float32))


def add_noise(x, npw, spw, snr_db, gen=None):
    """Same idea as the organizers' degrade(): add Gaussian noise to reach the target SNR."""
    added = spw / (10.0 ** (snr_db / 10.0)) - npw
    sigma = torch.sqrt(torch.clamp(added, min=0) / 2).unsqueeze(1)
    noise = torch.randn(x.shape, device=x.device, generator=gen) * sigma
    return torch.clamp(x + noise, -32768, 32767)


WIN = torch.hann_window(256).to(device)

def to_spec(x):
    """x: (B, 33600) interleaved I/Q -> (B, 1, 256 freq, 130 time) normalized log-spectrogram."""
    iq = torch.complex(x[:, 0::2], x[:, 1::2])
    S = torch.stft(iq, n_fft=256, hop_length=128, window=WIN, center=False,
                   return_complex=True, onesided=False)
    P = torch.fft.fftshift(S.abs() ** 2, dim=1)
    L = 10 * torch.log10(P + 1e-8)
    L = (L - L.mean((1, 2), keepdim=True)) / (L.std((1, 2), keepdim=True) + 1e-6)
    return L.unsqueeze(1)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

model = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128),
                      nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 7)).to(device)


@torch.no_grad()
def evaluate(X, y, npw, spw):
    model.eval()
    gen = torch.Generator(device=device); gen.manual_seed(123)
    per_snr = {}
    for snr in SNRS:
        preds = []
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).to(device).float()
            nb = torch.from_numpy(npw[i:i + 256]).to(device)
            sb = torch.from_numpy(spw[i:i + 256]).to(device)
            xb = add_noise(xb, nb, sb, float(snr), gen)
            preds.append(model(to_spec(xb)).argmax(1).cpu().numpy())
        preds = np.concatenate(preds)
        recalls = [(preds[y == c] == c).mean() for c in range(7) if (y == c).any()]
        per_snr[snr] = float(np.mean(recalls))
    return per_snr


if __name__ == "__main__":
    with open("split_manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    t0 = time.time()
    print("loading data...")
    Xtr, ytr, ntr, str_ = load_split([r for r in rows if r["split"] == "train"], N_TRAIN_PER_FILE)
    Xva, yva, nva, sva = load_split([r for r in rows if r["split"] == "validation"], N_VAL_PER_FILE)
    print(f"train {len(Xtr)}  val {len(Xva)}  ({time.time() - t0:.0f}s)")
    print("train class counts:", dict(zip(CLASSES, np.bincount(ytr, minlength=7).tolist())))

    counts = np.bincount(ytr, minlength=7)
    weights = torch.tensor(counts.sum() / (7 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    snr_t = torch.tensor(SNRS, dtype=torch.float32, device=device)

    best = -1
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = rng.permutation(len(Xtr))
        total, t1 = 0.0, time.time()
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb = torch.from_numpy(Xtr[idx]).to(device).float()
            nb = torch.from_numpy(ntr[idx]).to(device)
            sb = torch.from_numpy(str_[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            snr = snr_t[torch.randint(0, len(SNRS), (len(idx),), device=device)]  # random SNR per pulse
            xb = add_noise(xb, nb, sb, snr.unsqueeze(1).squeeze(1)) if False else add_noise(xb, nb, sb, snr)
            loss = loss_fn(model(to_spec(xb)), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        res = evaluate(Xva, yva, nva, sva)
        mean_ba = float(np.mean(list(res.values())))
        print(f"epoch {epoch}: loss {total / len(perm):.3f}  val mean balanced acc {mean_ba:.3f}  ({time.time() - t1:.0f}s)")
        print("   per SNR:", {k: round(v, 2) for k, v in res.items()})
        if mean_ba > best:
            best = mean_ba
            torch.save(model.state_dict(), "best_trackB.pt")
            json.dump(dict(track="B", n_fft=256, hop=128, window="hann", snrs=SNRS, seed=SEED,
                           train_per_file=N_TRAIN_PER_FILE, val_score=best, val_per_snr=res),
                      open("config_trackB.json", "w"), indent=2)
    print("best val mean balanced accuracy:", best)