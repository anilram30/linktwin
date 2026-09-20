# linktwin — predictive digital twin

**Predicts whether a complete Ethernet harness will pass — and what the receiver actually sees — before the harness exists.**

[![CI](https://github.com/anilram30/linktwin/actions/workflows/ci.yml/badge.svg)](https://github.com/anilram30/linktwin/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)](tests/)
[![Report](https://img.shields.io/badge/report-19%20pages-informational)](docs/report.pdf)

> **Part of the [HF cable toolchain](https://github.com/anilram30/hf-cable-toolchain)** — seven packages that take a high-frequency cable from a raw measurement to a predicted Ethernet link.
> 
> [A · cablecheck](https://github.com/anilram30/cablecheck)  ·  [B · labauto](https://github.com/anilram30/labauto)  ·  [C · shieldeval](https://github.com/anilram30/shieldeval)  ·  [D · zprofile](https://github.com/anilram30/zprofile)  ·  [E · cableanalytics](https://github.com/anilram30/cableanalytics)  ·  [F · labplatform](https://github.com/anilram30/labplatform)  ·  **G · linktwin**

---

## The problem it solves

This is what the rest of the toolchain is for. A harness engineer has to commit to a cable, a length, a number of connectors and a routing temperature months before the first physical sample exists. `linktwin` assembles a virtual link from measured cable data, from a cable that exists only as a production record, and from connector models, then answers the questions that actually decide the design: does it pass the standard at 85 °C and at 125 °C, how far can the cable be stretched, how many connectors can it take, what does the eye diagram look like at the receiver, and — given the tolerances — what is the probability that it passes at all.

## At a glance

|  |  |
|---|---|
| **Takes** | A harness description in TOML, and a cable taken from a measurement, a laboratory archive or a production record |
| **Produces** | Link-segment verdict with margins, the eye diagram at the receiver, maximum reach, connector budget and the probability of passing |
| **Checked against** | Ground truth from a synthesiser the twin never sees, and a bit-level symbol simulation of the eye |
| **Technical report** | [`docs/report.pdf`](docs/report.pdf) — 19 pages, 29 references, every method stated with its mathematics and its limitations |
| **Tests** | 21, run against Python 3.11, 3.12 and 3.13 on every push |
| **Data** | Entirely synthetic. No proprietary or customer measurements are used anywhere in this toolchain. |

## Install

Python 3.11 or newer.

```sh
pip install "git+https://github.com/anilram30/cablecheck.git" \
            "git+https://github.com/anilram30/linktwin.git"
```

Optional — `labauto`, `cableanalytics` unlock extra capability, and the package works without them:

```sh
pip install "git+https://github.com/anilram30/labauto.git" "git+https://github.com/anilram30/cableanalytics.git"
```

---

## What it does

The twin chains what the other projects produce — a measured
cable (A/B/F), a cable that exists only as an extrusion-line record (E), connector and PCB models —
into a complete link between two PHYs, and answers the questions a harness designer asks *before the
harness exists*:

* Does this link pass the link-segment limits (1000BASE-T1 and others, project A's limit files) at
  85 °C, at 125 °C — and by how much, at which frequency, because of which element?
* What does the receiver see: the pulse response, the worst-case (peak-distortion) eye, the eye after
  the PHY's FFE/DFE, the eye with noise at BER 10⁻¹² — checked against a bit-level simulation?
* How long can the cable be? How many inline connectors can it take? Up to what temperature?
* With the tolerances of cable, connectors, cutting and temperature — and, for a production-predicted
  cable, project E's prediction interval — what is the probability of passing, and which parameter
  dominates (Monte Carlo and tornado)?

Everything is synthetic: cables come from project A's synthesiser (with VNA noise), archives from
project B's simulated laboratory, production records from project E's data set. No real data.

## Install

```sh
pip install cablecheck_projectA_v0.1.1.zip     # required: limits, mixed-mode, cascade, synthesiser
pip install ./linktwin                           # this package
pip install labauto_... cableanalytics_...       # optional: archive and production cable sources
```

Python ≥ 3.11; numpy, scipy, matplotlib.

## Ten-minute tour

```sh
linktwin demo demo_out                    # example measurement + harnesses, everything, figures, index.html
linktwin evaluate examples/camera-link.toml
linktwin eye examples/camera-link.toml --fig eye.png
linktwin maxlength examples/camera-link.toml --temperatures 23,85,125 --min-eye-mv 100
linktwin limiting examples/camera-link.toml
linktwin connectors examples/camera-link.toml
linktwin montecarlo examples/camera-link.toml -n 200 --fig mc.png
linktwin sensitivity examples/camera-link.toml --fig tornado.png
linktwin validate --out validation          # the twin against the synthesiser it never saw
```

A harness is a TOML file: elements in order from transmitter to receiver, a standard, a PHY, a
temperature, tolerances. Cable elements name their source:

```toml
[[element]]
type = "cable"   name = "cable-1"   length_m = 4.0
source = "touchstone"   path = "pairA_10m_23C.s4p"   measured_length_m = 10   measured_at_c = 23
# source = "archive"      job_dir = "archive-job"            # a labauto job: length, temperature, trust, hashes from the sidecar
# source = "production"   dataset = "production_dataset.csv"  sample = "S0005"   # project E predicts a, b, Z (+ intervals)
# source = "coefficients" a = 1.66e-5  b = 2.0e-10  nvp = 0.68  z_diff = 100      # a data-sheet cable
```

## What is inside

| module | what it does |
|---|---|
| `mtl.py` | line/ABCD algebra, γ and Zc extraction from a measured line (branch by continuity), the causal propagation model α = (c0 + a√f + bf)/8.686, β = 2πf/v + a√f/8.686, closed-form two-conductor MTL chain matrix |
| `elements.py` | `CableSegment` (measured → per-metre model at any length/temperature, hybrid: causal fit + measured structure; or from coefficients), `Connector` (mismatched MTL stub with contact asymmetry, pin L, pad C), `PcbTrace` |
| `link.py` | cascade (cablecheck wave-chain, all four ports), mismatched transfer function with PHY Γ, link-segment verdict through project A's `compute_quantities`/`evaluate` |
| `eye.py` | Gaussian-edged PAM pulse, cursor window that follows the pulse tail, peak-distortion eye (raw / FFE+DFE), noise at BER 10⁻¹², bit-level simulation as validation |
| `uncertainty.py` | tolerances, Monte Carlo (P(pass), margin and eye percentiles, per-quantity pass probability), one-at-a-time tornado |
| `inverse.py` | maximum length (with non-monotonic pass/fail handled), connector budget, limiting element, temperature ceiling |
| `harness.py` | TOML harness; cable sources touchstone / archive (with fixture removal and trust) / production (project E) / coefficients |
| `validate.py` | the twin against the synthesiser: length and temperature scaling (both temperature conventions), extrapolation beyond the band, cascade vs truth harness, PDA vs simulation |
| `figures.py`, `demo.py`, `cli.py` | figures, the demonstration, the command line |

## Validation summary (from `linktwin validate`)

From one noisy 10 m measurement at 23 °C the twin predicts 3–25 m at 23–125 °C with an insertion-loss
error ≤ 0.1 dB over 5–600 MHz (0.09 dB at 25 m / 125 °C, of 16.7 dB), extrapolates to 2.5 GHz with
≤ 0.04 dB error, gives the same verdict as a harness built from the synthesiser's true pieces with the
headline margin within 0.002 dB, and its worst-case eye is a tight lower bound on the bit-level
simulation (420 vs 421 mV equalised at 5 m; 311 vs 311 mV at 15 m). Return-loss *pattern* of another
physical piece is not predictable; the twin carries the measured piece's impedance structure and its
RL minima land within about 3 dB.

## Temperature law

Copper's skin-effect resistance scales with √ρ(T); the twin therefore uses a(T) = a√(1 + αΔT), as project
E's physics does. The synthetic instrument stack of projects A/B keeps R_s at its 20 °C value (only R_dc
and tan δ move). `skin_temperature=False` on a cable model reproduces that convention; `validate` runs
both and the twin matches either truth to ≤ 0.1 dB.

## Report

`docs/report.pdf` — the mathematics (MTL, extraction, causal model, connector model, cascade, mismatched
transfer, PAM3 pulse, PDA, FFE/DFE, Monte Carlo, inverse problems), the validation, and the limitations.

## Tests

`pytest` — 21 tests: line algebra round trips, causality (Hilbert pair), matched-line delay, self-
reproduction, length/temperature scaling under both conventions, extrapolation, connector passivity /
reciprocity / mode conversion, cascade consistency, verdict rows, twin-vs-truth harness, pulse causality
and delay, PDA bound, eye monotonicity, inverse tools, Monte Carlo and sensitivity, harness loading and
every CLI command.
---

## Contributing

Bug reports, questions about the methods, and pull requests are all welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). Numerical changes need a numerical test, and a change to a method
is also a change to `docs/report.md`.

## Licence and attribution

MIT — see [LICENSE](LICENSE). Author: Sreeram Anil.

Built with AI assistance; the commit history records it. The engineering decisions, the validation
strategy and the limitations stated in the report are the substance of the work.

Part of the **[HF cable toolchain](https://github.com/anilram30/hf-cable-toolchain)** · [Report an issue](https://github.com/anilram30/linktwin/issues) ·
[Changelog](CHANGELOG.md)
