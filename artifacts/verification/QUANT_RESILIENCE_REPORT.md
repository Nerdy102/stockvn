# QUANT_RESILIENCE_REPORT

## CP events examples
- Included synthetic OCPD variance/impact-shift detections in unit tests.

## Regime timeline
- Regime inference + hysteresis + CP-aware toggles are covered by deterministic tests.

## Uncertainty coverage summary
- Rolling coverage / width / ECE metrics utility added and tested.

## Expert weight evolution sample
- 60m multiplicative-update path tested with deterministic outputs and caps.

## Portfolio action logs
- (Pending richer integration wiring in service layer.)

## TCA calibration params sample
- Robust Huber regression params + deterministic param hash record added and tested.

## Timing benchmarks
- 500-symbol 60m alpha update microbench test asserts runtime < 3s.
