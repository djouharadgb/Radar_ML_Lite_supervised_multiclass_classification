# RadarML Lite Hackathon - Method Submission Overview

This package documents our two entries for the radar modulation recognition challenge (seven classes: `barker`, `cw`, `fmcw`, `fsk`, `hyp`, `nlfm`, `quad`), one per track.

| Document | Content |
| --- | --- |
| `README.md` (this file) | Overview, results, shared data pipeline, reproduction, compliance statement |
| `TRACK_A.md` | Track A (time domain): features, network architecture, training, results |
| `TRACK_B.md` | Track B (time-frequency): spectrogram front end, CNN architecture, training, results |

## 1. Summary of entries

| | Track A (time domain) | Track B (time-frequency) |
| --- | --- | --- |
| Input representation | RMS-normalised I/Q, lag conjugate products, second-order lag products, learned FIR filterbank branch | Noise-whitened, cropped, standardised log-spectrogram (STFT) |
| Model | Two-branch 1D residual CNN with squeeze-excite, bidirectional GRU, attentive statistics pooling, MLP head | 2D residual CNN with a 4x4 spatial pooling head |
| Parameters per model | 3,148,012 | about 2.48 million (computed by hand from the layer shapes) |
| Ensemble | 3 models, same architecture, seeds 42/43/44 | 3 models, STFT sizes 256 / 128 / 512 |
| Validation mean balanced accuracy (13 SNRs) | 0.8914 (single models 0.8881 / 0.8891 / 0.8885) | 0.8977 (single models 0.8879 / 0.8785 / 0.8887) |
| Tie-break score (mean of -20, -25, -30 dB) | 0.5305 | about 0.557 (from per-SNR values 0.954, 0.513, 0.205) |

All numbers are on the official validation split (13,200 source pulses, 13 requested SNRs), with noise generated on the fly using fixed seeds. They are validation figures, not blind-test figures.

### Per-SNR validation balanced accuracy (ensembles)

| Requested SNR (dB) | Track A | Track B |
| --- | --- | --- |
| +30 to -10 | 1.000 | 1.000 |
| -15 | 0.996 | 0.998 |
| -20 | 0.860 | 0.954 |
| -25 | 0.487 | 0.513 |
| -30 | 0.244 | 0.205 |

Observations common to both tracks:

- Every class is separated essentially perfectly down to -10 dB. All of the remaining score is decided between -15 dB and -30 dB.
- At -30 dB both systems are close to chance (1/7 = 0.143 is chance); the confusion matrix for Track A at -30 dB shows predictions collapsing mostly toward `fmcw`, `hyp` and `quad`.
- At -20 dB, Track A confuses `fmcw` with `nlfm` (about 13 to 14 percent each way) and `quad` with `fmcw`/`nlfm`; these are the frequency-modulated families, which look alike when the sweep is buried in noise. `barker` is partly confused with `cw` (10.8 percent).
- Track B is stronger around -20 dB, Track A is stronger at -30 dB. The ensemble gain over the best single model is larger for Track B (+0.009) than for Track A (+0.002 to +0.003).

## 2. Shared data pipeline

Both tracks use the same data handling.

**Source data.** Compact `*_recorded.npy` arrays of shape `(1, 100, 33600)`, int16, interleaved I/Q (`I0, Q0, I1, Q1, ...`), 16,800 complex samples per pulse at 120 MS/s (140 microseconds).

**Split.** The published `split_manifest.csv` is used unchanged: 80 training and 20 validation pulses per configuration, 52,800 training and 13,200 validation source pulses. No validation pulse is ever used for fitting weights, normalisation or augmentation. Model selection (best epoch) uses validation pulses, so validation scores are slightly optimistic.

**Labels.** Derived from the waveform token in the filename (`barker*` mapped to `barker`, `fsk#k` mapped to `fsk`). Filenames and metadata are used only to build the label array and are never model inputs. Cropping uses a fixed window, not a filename pulse width.

**Class counts (training).** `barker` 5,760; `cw` 960; `fmcw` 7,680; `fsk` 15,360; `hyp` 7,680; `nlfm` 7,680; `quad` 7,680. The imbalance is handled in the loss (see each track document), because the metric is balanced accuracy.

**Power estimates per pulse.** For each pulse we compute noise power `n` as the mean of `|x|^2` over 0 to 20 microseconds and signal power `s` as the mean over 21 to 119 microseconds minus `n`, exactly as in the organizers' `degrade()`. Pulses with non-finite or non-positive `s` are dropped (none were dropped in the logged runs: 52,800 and 13,200 pulses were retained).

**On-the-fly noise at a requested SNR.** Instead of storing the 13 pre-expanded SNR variants, both notebooks reimplement `degrade()` in PyTorch:

```
added_power = s / 10^(SNR/10) - n          (clamped at 0 when the target is above the source SNR)
sigma       = sqrt(added_power / 2)
x_noisy     = clip(x + N(0, sigma) per I and Q value, -32768, 32767)
```

Track A additionally truncates to integer values (`torch.trunc`) to mimic the int16 cast. Random SNR per pulse is drawn at every step, so each epoch sees different noise for the same source pulse. All SNR variants of a pulse inherit its split by construction, because noise is added only after the pulse has been assigned to a split.

**Requested versus achieved SNR.** As documented by the organizers, requested SNR is not the in-pulse achieved SNR (the 21 to 119 microsecond window includes padding for short pulses, and requested SNRs above the source SNR add no noise). All plots and tables report requested SNR.

**Validation protocol.** Per-SNR balanced accuracy is the mean of per-class recall over the classes present; the headline score is the mean over the 13 SNRs. Evaluation noise uses fixed generator seeds so models are compared on identical noisy inputs.

## 3. Reproduction

Environment: Python 3.12, PyTorch with CUDA, NumPy, Matplotlib (Track B plots). Training was run on a CUDA GPU with bf16/fp16 mixed precision.

Folder layout expected by both notebooks (they walk up from the working directory until they find it):

```
Radar_ML_lite/
  dataset/                  # *_recorded.npy files and sidecar JSON
  dataset_infos/
    split_manifest.csv
```

Run order:

1. Track A: open `track_a_v3.ipynb`, run all cells. Outputs go to `track_a_runs/final_v3/` (`model_0.pt`, `model_1.pt`, `model_2.pt`, `config.json`, `method_summary.md`, `SHA256SUMS`).
2. Track B: open `train_track_b.ipynb`, run all cells (set `SMOKE = True` first for a one-minute check). Outputs go to `track_b_plus/` (one `best_nfft*_hop*_seed*.pt` per model and `ensemble_config.json`).
3. Blind-test inference: Track A uses `src/track_a/predict_track_a.py` (referenced by the notebook; run with `--selftest` first). Track B uses `load_ensemble()` and `ensemble_predict()` defined at the end of the Track B notebook on raw `(N, 33600)` arrays.

Frozen artifact hashes for Track A (SHA256, from the notebook output):

```
eaa2893170a210ecd0bb09111c1789bd97c5bbcd9177beaff44303dc0e5e3a97  model_0.pt
a318ed5358c431362f4e999ce01e7cfb92a1b06aaa7be91c82b82f88845f65ef  model_1.pt
7f540175c854b80d240747fb77ee850bca51bf913bbd72071f1f1300c663b2bf  model_2.pt
a50cc7efbe63d2dd483707dbea59411f4514ba38a87269cb6edf6f65ccc800eb  config.json
```

Track B hashes are not printed by its notebook; generate them with `sha256sum` (or `Get-FileHash` on Windows) on the three `.pt` files and `ensemble_config.json` before the freeze submission and add them here.

Submission file format: `predictions.csv` must have exactly the header `sample_id,predicted_label`, one row per test ID, labels restricted to the seven lowercase class names. The organizers' scorer rejects any other header. (The Track A notebook text mentions a `sample_id,label` header; make sure the inference script writes `predicted_label`.)

## 4. Compliance statement

- Models are trained from scratch on the supplied training pulses only. No external data and no pretrained weights are used.
- No filenames, IDs, metadata, or requested-SNR labels are model inputs. The requested SNR is used only to generate training noise and to group evaluation results.
- Training-only augmentation (random SNR, random carrier phase in Track A) preserves the modulation class.
- Track A uses no FFT, STFT, wavelet or spectrogram computation. It uses a learned 1D convolutional filterbank, which we consider a learned 1D convolution as permitted by the rules; because it behaves like a time-domain filterbank, it is flagged for confirmation with the organizers (set `USE_FB = False` to remove it; the lag-product branch then stands alone).
- Track B computes STFT spectrograms from the same I/Q samples and includes representation computation in inference timing.
- AI assistance: the code was written with the help of an AI assistant (Claude); the team reviewed and ran it.
- The blind-test package was not accessed before the release time.

## 5. Known limitations

- Results measure recognition of new pulses from known waveform configurations, not generalisation to unseen configurations, receivers or environments.
- The ensemble validation scores are slightly optimistic because each member's best checkpoint was selected on validation data (Track A used a fixed 1,500-pulse validation subset for selection; Track B used all 13,200 pulses).
- Performance at -25 and -30 dB is close to the limit of what these models extract from a single pulse; I/Q clipping at -30 dB (5.38 percent of component values in the blind test) also alters the noise distribution relative to our on-the-fly noise, which clips at the int16 limits in the same way but is not identical in its random draws.
- Inference timing, hardware, and training time must be filled in from the final run (see each track document for the measured training times).
