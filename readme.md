# LesZivés – SynapsON Radar ML Lite Challenge

Radar pulse **modulation classification** from raw I/Q samples, built for the SynapsON Radar ML Lite Challenge (Kaggle datathon).

**Team:** LesZivés (4 members)
**Kaggle teams:** `LesZivés_time` (Track A) · `LesZivés_freq` (Track B)

---

## 1. Task

Each example is one radar pulse: 16,800 complex samples at 120 MS/s (int16 interleaved I/Q). The goal is to predict one of 7 modulations:

`barker, cw, fmcw, fsk, hyp, nlfm, quad`

- **Metric:** balanced accuracy computed per requested SNR (+30 … −30 dB, 13 levels), then averaged.
- **Tie-break:** mean balanced accuracy at −20 / −25 / −30 dB.
- **Test set:** 171,600 rows (13,200 pulses × 13 SNR levels).
- **Submission format:** CSV with header exactly `sample_id,label`.

| Track | Allowed representation | Kaggle team |
|-------|------------------------|-------------|
| **A** | Time-domain only (no FFT / STFT / wavelet / spectrogram) | `LesZivés_time` |
| **B** | Time-frequency representations | `LesZivés_freq` |

---

## 2. Results

| Track | Model | Validation | Leaderboard |
|-------|-------|-----------|-------------|
| A | Time-domain CNN + bi-GRU, 3-model ensemble (v3) | **0.8914** (ensemble, 13,200 val pulses) | see Kaggle |
| B | STFT log-spectrogram 2D ResNet, 3-model ensemble | ≈ 0.89 | **0.896** |

Track A self-test (synthetic degradation on validation, ensemble v3):

| SNR (dB) | 0 | −10 | −15 | −20 | −25 |
|----------|---|-----|-----|-----|-----|
| Balanced acc. | 1.000 | 1.000 | 0.996 | 0.854 | 0.506 |

Most of the remaining error is at SNR ≤ −20 dB, which is where the tie-break lives.

---

## 3. Repository layout

```
Radar_ML_lite/
├── src/
│   ├── track_a/
│   │   └── predict_track_a_v3.py      # Track A inference -> submission CSV
│   └── track_b/
│       └── predict_track_b.py         # Track B inference -> submission CSV
├── notebooks/
│   ├── track_a_v3.ipynb               # Track A training (time-domain)
│   └── train_b_plus.ipynb             # Track B training (STFT ensemble)
├── track_a_runs/final_v3/             # Track A weights: config.json, model_0..2.pt
├── track_b_plus/                      # Track B weights: ensemble_config.json, best_nfft*_hop*_seed*.pt
├── submissions/                       # generated CSVs (prediction_track_a_v3.csv, predictions_track_b.csv)
├── EVENT_RULES.md                     # organizers' rules
├── participant_loader.py              # organizers' loader
├── score_submission.py                # organizers' scorer
└── requirements.txt
```

The dataset (training / validation / blind test) is provided by the organizers and is **not** included in this repository.

---

## 4. Setup

```powershell
cd Radar_ML_lite
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# + a CUDA build of PyTorch if a GPU is available
```

Linux / macOS: `source .venv/bin/activate`.

---

## 5. Generate the submission files

Run from the `Radar_ML_lite` folder with the virtual environment active.

### Track A (time-domain) – `LesZivés_time`

```powershell
# 1) sanity check on synthetic noisy validation data (writes no CSV)
python src\track_a\predict_track_a_v3.py --selftest

# 2) prediction on the blind test set
python src\track_a\predict_track_a_v3.py
```

Output: `submissions\prediction_track_a_v3.csv` and `submissions\prediction_track_a_v3_run.json` (weights hash, device, timing).

Optional flags: `--out <path>`, `--blind-dir <path>`, `--batch <n>`, `--verify-all` (hash every `.npy` against `checksums.json`).

### Track B (time-frequency) – `LesZivés_freq`

```powershell
python src\track_b\predict_track_b.py --selftest
python src\track_b\predict_track_b.py
```

Output: `submissions\predictions_track_b.csv`.

### Checks performed automatically

Both scripts:
- locate the blind-test folder and verify its checksums;
- write rows in the order of `submission_template.csv`;
- check that the ID set equals the template's, the header is `sample_id,label`, and every label is one of the 7 classes;
- re-read the written file to validate it.

Expected: **171,600 rows + 1 header**, all 7 classes present.

```powershell
dir submissions
Get-Content submissions\prediction_track_a_v3.csv -TotalCount 3
```

---

## 6. Method

### Shared ideas
- Per-pulse RMS normalization.
- Signal-window crop (21–119 µs, 11,760 samples) to drop the leading noise-only part and the tail.
- **SNR emulation during training:** a torch port of the organizers' `degrade()` (noise scaled to the requested SNR, int16 clipping) so training sees the same noise as the test.
- Random carrier-phase augmentation.
- **SNR curriculum:** first epochs at easy SNRs, then 50% of draws in [−30, −10] dB, where the metric is decided.
- Loss tuned for balanced accuracy (logit adjustment, label smoothing); AdamW + OneCycle; EMA of weights; bf16 autocast.
- Ensemble of 3 independently seeded models, softmax averaged.

### Track A – time domain only
- Input channels: I, Q, plus **lag conjugate products** `z[n]·conj(z[n−k])` for k ∈ {1,2,4,8,16,32} and second-order products for lag pairs (4,4), (8,8), (16,16). These are sample-domain operations (no transform).
- Backbone: stem conv → 6 residual squeeze-and-excitation stages (stride 2).
- A learned 1D conv filter-bank branch (`Conv1d(2→128, k=256, stride=16)`, magnitude of 64 complex filters, 3 residual blocks) concatenated with the backbone features.
- Bi-GRU → attentive statistics pooling → MLP head.
- 80 epochs, seeds 42 / 43 / 44.

### Track B – time-frequency
- STFT log-magnitude spectrogram, whitened with the noise floor estimated on the first 20 µs, then cropped to the signal window.
- Residual 2D CNN with a 4×4 pooled head.
- Ensemble of 3 models with different STFT settings (`n_fft` = 256 / 128 / 512).

---

## 7. Rule compliance

- Trained **from scratch on the supplied training pulses only**; validation pulses were never used for training.
- No model change or test-driven adaptation after the freeze.
- No sample IDs, row numbers or requested SNR used as features.
- Each test sample is classified independently of other test samples (no cross-example linking).
- One final entry per track, submitted from the matching Kaggle team.
- Track A uses no FFT / STFT / wavelet / spectrogram call anywhere in its pipeline.

