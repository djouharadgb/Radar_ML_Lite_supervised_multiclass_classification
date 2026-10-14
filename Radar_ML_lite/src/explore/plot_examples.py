"""Draw one training pulse per class to build intuition (not part of any submission).
Columns: magnitude at +30 dB | instantaneous frequency at +30 dB | instantaneous frequency at -10 dB.
Run:  python src/explore/plot_examples.py     (output: plots/class_examples.png)
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "dataset"
SPLIT_CSV = ROOT / "dataset_infos" / "split_manifest.csv"
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

FS = 120e6
CLASSES = ["barker", "cw", "fmcw", "fsk", "hyp", "nlfm", "quad"]
N0, S0, S1 = int(20e-6 * FS), int(21e-6 * FS), int(119e-6 * FS)
rng = np.random.default_rng(0)


def get_label(name):
    t = name.split("_")[1].split("#")[0].lower()
    return "barker" if t.startswith("barker") else t


def load_iq(file, idx):
    raw = np.load(DATA_DIR / file, mmap_mode="r")[0, idx]
    return raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)


def at_snr(iq, snr_db):
    n = np.mean(np.abs(iq[:N0]) ** 2)
    s = np.mean(np.abs(iq[S0:S1]) ** 2) - n
    sigma = np.sqrt(max(0.0, s / 10 ** (snr_db / 10) - n) / 2)
    return iq + sigma * (rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size))


def inst_freq_mhz(iq, smooth=64):
    f = np.angle(iq[1:] * np.conj(iq[:-1])) * FS / (2 * np.pi) / 1e6
    return np.convolve(f, np.ones(smooth) / smooth, mode="same")


examples = {}
with open(SPLIT_CSV, newline="") as fh:
    for r in csv.DictReader(fh):
        c = get_label(r["file"])
        if r["split"] == "train" and c not in examples:
            examples[c] = (r["file"], int(r["pulse_index"]))

t = np.arange(16800) / FS * 1e6
fig, axes = plt.subplots(7, 3, figsize=(15, 17), sharex=True)
for row, c in enumerate(CLASSES):
    file, idx = examples[c]
    iq = load_iq(file, idx)
    clean, noisy = at_snr(iq, 30), at_snr(iq, -10)
    axes[row, 0].plot(t, np.abs(clean), lw=.6)
    axes[row, 1].plot(t[1:], inst_freq_mhz(clean), lw=.8, color="C1")
    axes[row, 2].plot(t[1:], inst_freq_mhz(noisy), lw=.8, color="C3")
    axes[row, 0].set_ylabel(c, fontsize=13, rotation=0, labelpad=30)
for ax, title in zip(axes[0], ["Magnitude (+30 dB)", "Frequency in MHz (+30 dB)", "Frequency in MHz (-10 dB)"]):
    ax.set_title(title)
for ax in axes[-1]:
    ax.set_xlabel("time (microseconds)")
fig.tight_layout()
fig.savefig(OUT / "class_examples.png", dpi=110)
print("saved", OUT / "class_examples.png")