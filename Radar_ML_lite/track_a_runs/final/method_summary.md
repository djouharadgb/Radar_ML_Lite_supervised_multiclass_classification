# Track A method summary
Strictly time-domain processing (no frequency-domain or time-frequency transform of any kind). Input: complex I/Q of the 21-119 us window, divided by per-pulse RMS.
Channels: I, Q, and real/imag of z[n]*conj(z[n-k]) for k in [1, 4, 16].
Model: learned 1D conv stem + residual SE conv blocks + bidirectional GRU + attentive statistics pooling + MLP.
Training: official 80/20 split, on-the-fly Gaussian noise at random SNR (low SNR oversampled), random carrier phase, logit-adjusted
label-smoothed CE (balanced accuracy), AdamW + OneCycle, EMA weights, bf16 autocast.
Validation mean balanced accuracy over the 13 SNRs: 0.8656.
AI assistance: the code was written with help from an AI assistant (Claude); the team reviewed and ran it.
