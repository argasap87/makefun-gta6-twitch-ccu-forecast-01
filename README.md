# GTA VI launch-day Twitch forecast

Entry for make.fun `gta6-twitch-ccu-forecast-01`: forecast how many people
watch Grand Theft Auto VI live on Twitch in each hour (2–24) of its release
day, 2026-11-19 UTC, as five quantiles per hour, from the first hour's print.

## Program

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

## Approach

For each hour `h`, forecast the log-ratio `ln(viewers_h / viewers_1)` from
the case's comparables (past big launches' hourly curves):

- **Training set**: comparables with a usable release day that look like a
  rolling midnight-local launch (hour 1 ≥ 5% of the day's max) and are not
  flagged early-access-contaminated or weak. Console-only launches get
  double weight — GTA VI is PS5/Xbox-only.
- **Conditional median**: per hour, a weighted least-squares line of the
  log-ratio on log hour-1 viewers. Big hour-1 launches stay flat; small
  hour-1 launches climb steeply into the US evening (late-hour correlation
  ≈ −0.8). Slope clamped to ±0.6; the predictor is clamped to the training
  range so a record-breaking GTA VI hour 1 extrapolates as the biggest
  launch seen, not beyond it.
- **Spread**: empirical quantiles of that hour's regression residuals,
  centred on the residual median — an honest, data-driven band.
- Fallbacks (rolling-only, then any usable launch, then the persistence
  baseline) keep the output valid if the filter leaves too few launches.
  Fully deterministic: no randomness, sorted iteration.

## Validation (leave-one-out over the comparables)

Each held-out launch's own hourly series is ground truth; the model is
trained on the other launches only.

| Universe | Model pinball | Baseline pinball | Typical miss | 90% coverage |
|---|---|---|---|---|
| Rolling + clean (n=19) | **0.263** | 0.403 | 45% | 0.83 |
| Rolling (n=29) | **0.273** | 0.360 | 49% | 0.81 |
| Console-only (n=7) | **0.164** | 0.344 | 26% | 0.93 |

House baseline on the practice cases: 0.387. The model beats it clearly
in every comparable universe.

## Local run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
