import csv
import numpy as np

CLASSES = ["barker", "cw", "fmcw", "fsk", "hyp", "nlfm", "quad"]

def get_label(filename):
    token = filename.split("_")[1].split("#")[0].lower()
    if token.startswith("barker"):
        token = "barker"
    return token

# read the split file
with open("split_manifest.csv", newline="") as f:
    rows = list(csv.DictReader(f))

train_rows = [r for r in rows if r["split"] == "train"]
val_rows = [r for r in rows if r["split"] == "validation"]
print("train:", len(train_rows), "validation:", len(val_rows))

# check the labels
print(sorted({get_label(r["file"]) for r in rows}))

# load one training example
r = train_rows[0]
arr = np.load(r["file"], mmap_mode="r")
raw = arr[0, int(r["pulse_index"])]
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
y = CLASSES.index(get_label(r["file"]))
print("file:", r["file"])
print("iq shape:", iq.shape, "label:", y)