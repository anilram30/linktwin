# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-18

### Added
- First release: the full-link digital twin.
- Line algebra: ABCD/S conversions, γ and Z_c extraction from a measured line with branch selection by
  continuity along frequency, the causal per-metre model (c₀ + a√f + b·f loss with its skin-effect
  phase companion), and a closed-form two-conductor MTL chain matrix.
- Elements: measured cable to a per-metre model at any length and temperature (hybrid causal fit plus
  measured impedance structure, with the γ residual switchable off for time-domain work), cable from
  coefficients including `cableanalytics` predictions with design-physics NVP and DC floor, connector
  as a mismatched asymmetric MTL stub with pin inductance and pad capacitance, PCB trace.
- Link: four-port wave-chain cascade, link-segment verdict through `cablecheck`, mismatched PHY
  transfer function.
- Eye: Gaussian-edged PAM pulse, cursor window that follows the pulse tail, peak-distortion eye,
  zero-forcing FFE with an ideal DFE, noise at a bit-error ratio of 1e-12, and a bit-level simulation
  that validates the bound.
- Uncertainty: tolerances, Monte Carlo and a one-at-a-time tornado. Inverse problems: maximum length
  with non-monotonic pass/fail handled, connector budget, limiting element, temperature ceiling.
- Harness TOML with `touchstone`, `archive` (a `labauto` job, with fixture removal and trust),
  `production` (a `cableanalytics` prediction) and `coefficients` cable sources.
- CLI, demo, figures, validation against the synthesiser under both temperature conventions,
  21 tests and a 19-page technical report.
