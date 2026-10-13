# Recorded RadarML pulses

Each *_recorded.npy contains (1, pulses, 33600) interleaved int16 I/Q values at 120 MS/s. Original recorded noise remains; no synthetic noise has been added. Keep the JSON sidecars with the arrays.

Install Python and NumPy (`python -m pip install numpy`), then run from this folder:

```
python compact_dataset.py expand --input . --output ../expanded
```

Default expansion uses 13 requested SNRs (+30 to -30 dB), seed 42, and about 13 times the compact array storage. Add `--snrs 20 10 0 -10 -20` for a smaller sweep. Outputs match the organizer preprocessing for identical settings and NumPy environment. All variants of a source pulse must stay in the same train/validation/test split. This source-named dataset exposes labels; it is intended for an open workshop, not an anonymized blind competition.

Source: Ritchie, White and Hosford (2025), RadarML, https://doi.org/10.5522/04/30752767.v1. The included LICENSE covers preprocessing code (GPL-3.0); dataset licensing is separate; see the source record.
