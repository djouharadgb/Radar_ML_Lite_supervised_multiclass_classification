# Track A method summary
Strictly time-domain processing (no frequency-domain or time-frequency transform of any kind). Input: complex I/Q of the 21-119 us window, divided by per-pulse RMS.
Channels: I, Q, real/imag of z[n]*conj(z[n-k]) for k in [1, 2, 4, 8, 16, 32], and second-order products p_k[n]*conj(p_k[n-m]) for (k,m) in [(4, 4), (8, 8), (16, 16)]. Second branch: learned FIR filterbank (64 complex filters, length 256, random init, stride 16) followed by magnitude (all learned time-domain convolutions; no FFT/STFT/wavelet).
Model: two-branch learned 1D conv (lag-product branch + filterbank branch) + residual SE conv blocks + bidirectional GRU + attentive statistics pooling + MLP; ensemble of 3 models (seeds [42, 43, 44]), softmax averaged.
Training: official 80/20 split (training pulses only, 52,800), on-the-fly Gaussian noise at random SNR (curriculum over 12 epochs, then 50% of draws in [-30,-10] dB), random carrier phase,
logit-adjusted label-smoothed CE (balanced accuracy), AdamW + OneCycle, EMA weights, bf16 autocast, 80 epochs.
Validation mean balanced accuracy over the 13 SNRs: 0.8914; tie-break (-20/-25/-30 dB): 0.5305.
AI assistance: the code was written with help from an AI assistant (Claude); the team reviewed and ran it.
