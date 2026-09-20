"""
The demonstration: example data + harnesses, then everything the twin can do on them, with figures
and a JSON record, in one output folder.

    linktwin demo OUT [--archive-job DIR] [--production-dataset CSV] [--quick]

Example harnesses written (and shipped in linktwin/harnesses/):

    camera-link.toml        ECU -> 6 m + inline connector + 9 m -> camera, the cable from a 10 m Touchstone
                            measurement (project A's synthesiser plus VNA noise), evaluated at 85 °C
    coefficients-link.toml  the same harness from loss coefficients only (a data-sheet cable)
    archive-link.toml       the cable from a labauto archive job (project B) - written if a job is given
    production-link.toml    the cable from a production data set row (project E) - written if cableanalytics
                            is installed
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
from cablecheck.io import write_touchstone
from cablecheck.synth import PairSpec, synthesize_measurement

from . import __version__
from .eye import eye_analysis
from .figures import (
    fig_elements,
    fig_extrapolation,
    fig_link,
    fig_max_length,
    fig_montecarlo,
    fig_pulse_eye,
    fig_scaling,
    fig_tornado,
)
from .harness import load_harness
from .inverse import connector_budget, limiting_element, max_length, temperature_ceiling
from .uncertainty import monte_carlo, sensitivity
from .validate import truth_pair, validate_cascade, validate_eye, validate_scaling

__all__ = ["write_examples", "run_demo"]

CAMERA = """# ECU-A to camera: 4 m + 8 m of one cable type with an inline connector, 85 °C in the engine bay
[harness]
name = "ECU-A to camera"
standard = "1000base-t1-link-segment"
phy = "1000base-t1"
temperature_c = 85

[[element]]
type = "pcb"
name = "ecu-pcb"
length_m = 0.06
z_diff = 95

[[element]]
type = "connector"
name = "ecu-header"
z_diff = 90
length_m = 0.02
asym_c = 0.01

[[element]]
type = "cable"
name = "cable-1"
length_m = 4.0
source = "touchstone"
path = "pairA_10m_23C.s4p"
measured_length_m = 10
measured_at_c = 23
port_map = "A+near,A-near,A+far,A-far"

[[element]]
type = "connector"
name = "inline"
z_diff = 88
length_m = 0.03
asym_c = 0.02

[[element]]
type = "cable"
name = "cable-2"
length_m = 8.0
source = "touchstone"
path = "pairA_10m_23C.s4p"
measured_length_m = 10
measured_at_c = 23

[[element]]
type = "connector"
name = "camera-header"
z_diff = 90
length_m = 0.02
asym_c = 0.01

[[element]]
type = "pcb"
name = "camera-pcb"
length_m = 0.04
z_diff = 95

[tolerances]
cable_a_rel = 0.02
cable_b_rel = 0.05
cable_z_ohm = 1.5
cable_length_rel = 0.005
temperature_k = 3.0
conn_z_ohm = 5.0
conn_asym_c_abs = 0.01
"""

COEFFICIENTS = """# The same harness from a data sheet: loss coefficients, NVP and impedance only (no measurement)
[harness]
name = "ECU-A to camera (data-sheet cable)"
standard = "1000base-t1-link-segment"
phy = "1000base-t1"
temperature_c = 85

[[element]]
type = "pcb"
name = "ecu-pcb"
length_m = 0.06

[[element]]
type = "connector"
name = "ecu-header"

[[element]]
type = "cable"
name = "cable-1"
length_m = 4.0
source = "coefficients"
a = 1.66e-5
b = 2.0e-10
nvp = 0.68
z_diff = 100

[[element]]
type = "connector"
name = "inline"
z_diff = 88
asym_c = 0.02

[[element]]
type = "cable"
name = "cable-2"
length_m = 8.0
source = "coefficients"
a = 1.66e-5
b = 2.0e-10
nvp = 0.68
z_diff = 100

[[element]]
type = "connector"
name = "camera-header"

[[element]]
type = "pcb"
name = "camera-pcb"
length_m = 0.04
"""

ARCHIVE = """# The cable from a labauto archive job (project B): length, temperature, trust and hashes come from the sidecar
[harness]
name = "backbone from archive job"
standard = "1000base-t1-link-segment"
phy = "1000base-t1"
temperature_c = 105

[[element]]
type = "connector"
name = "header-a"

[[element]]
type = "cable"
name = "trunk"
length_m = 12.0
source = "archive"
job_dir = "archive-job"

[[element]]
type = "connector"
name = "header-b"
"""

PRODUCTION = """# A cable that was never measured: project E predicts a, b, Z from the extrusion-line record of sample S0005
[harness]
name = "link from production record S0005"
standard = "1000base-t1-link-segment"
phy = "1000base-t1"
temperature_c = 85

[[element]]
type = "connector"
name = "header-a"

[[element]]
type = "cable"
name = "S0005"
length_m = 15.0
source = "production"
dataset = "production_dataset.csv"
sample = "S0005"

[[element]]
type = "connector"
name = "inline"
z_diff = 88
asym_c = 0.02

[[element]]
type = "connector"
name = "header-b"
"""


def write_examples(out: str | Path, archive_job: str | Path | None = None, production_dataset: str | Path | None = None, seed: int = 3) -> dict:
    """Write the example measurement and the harness files; returns {name: path} of the harnesses written."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = np.linspace(1e6, 600e6, 1200)
    base = PairSpec(ripple_amp=0.004, roughness=0.003, seed=seed)
    truth = truth_pair(10.0, 23.0, f, base)
    meas = synthesize_measurement(truth, None, noise_db=-85.0, seed=1, name="pairA 10 m 23 C")
    write_touchstone(meas, out / "pairA_10m_23C.s4p", comments=["synthetic 10 m pair (cablecheck.synth) with VNA noise, 23 C"])
    written = {}
    (out / "camera-link.toml").write_text(CAMERA, encoding="utf-8"); written["camera-link"] = out / "camera-link.toml"
    (out / "coefficients-link.toml").write_text(COEFFICIENTS, encoding="utf-8"); written["coefficients-link"] = out / "coefficients-link.toml"
    if archive_job and Path(archive_job).exists():
        dst = out / "archive-job"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(archive_job, dst)
        (out / "archive-link.toml").write_text(ARCHIVE, encoding="utf-8"); written["archive-link"] = out / "archive-link.toml"
    if production_dataset and Path(production_dataset).exists():
        shutil.copy(production_dataset, out / "production_dataset.csv")
        try:
            import cableanalytics  # noqa: F401
            (out / "production-link.toml").write_text(PRODUCTION, encoding="utf-8"); written["production-link"] = out / "production-link.toml"
        except ImportError:
            pass
    return written


def run_demo(out: str | Path, archive_job=None, production_dataset=None, quick: bool = False, log=print) -> dict:
    out = Path(out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = write_examples(out, archive_job, production_dataset)
    record = {"linktwin": __version__, "harnesses": {}, "validation": {}, "figures": []}
    n_mc = 60 if quick else 200

    for key, path in written.items():
        log(f"== {key}")
        h = load_harness(path)
        link = h.link
        r = link.evaluate()
        rec = {"describe": h.describe(), "evaluation": r.to_dict()}
        log(f"   {r.verdict}: headline {r.headline} margin {r.headline_margin:+.2f} dB; IL {', '.join(f'{k} {v:.2f}' for k, v in r.il_db.items())} dB")
        record["figures"].append(fig_link(link, figs / f"{key}_link.png"))
        record["figures"].append(fig_elements(link, figs / f"{key}_elements.png"))
        eye = eye_analysis(link, h.phy, simulate=True, n_symbols=3000 if quick else 6000)
        rec["eye"] = eye.to_dict()
        log(f"   eye {eye.phy}: raw {eye.height_raw_v * 1e3:.0f} mV (sim {eye.sim_height_raw_v * 1e3:.0f}), equalised {eye.height_eq_v * 1e3:.0f} mV (sim {eye.sim_height_eq_v * 1e3:.0f}), with noise {eye.height_noise_v * 1e3:.0f} mV")
        record["figures"].append(fig_pulse_eye(eye, figs / f"{key}_eye.png"))
        rec["limiting_element"] = limiting_element(link)
        log("   limiting: " + ", ".join(f"{d['element']} {d['gain_db']:+.2f} dB" for d in rec["limiting_element"][:3]))
        curve = []
        for T in ((23.0, 85.0, 125.0) if quick else (23.0, 60.0, 85.0, 105.0, 125.0)):
            m = max_length(link, T)
            me = max_length(link, T, phy=h.phy, min_eye_v=0.10)
            curve.append({"temperature_c": T, "max_length_m": m["max_length_m"], "limited_by": m["limited_by"], "max_length_eye_m": me["max_length_m"]})
        rec["max_length_vs_temperature"] = curve
        log("   max length: " + ", ".join(f"{c['temperature_c']:.0f} °C {c['max_length_m']:.1f} m ({c['limited_by']})" for c in curve))
        record["figures"].append(fig_max_length(curve, figs / f"{key}_maxlength.png", f"{h.name}: maximum reach"))
        rec["temperature_ceiling"] = temperature_ceiling(link)
        from .elements import Connector
        rec["connector_budget"] = connector_budget(link, Connector(z_diff=88.0, asym_c=0.02, name="inline"), max_n=6, temperature_c=h.temperature_c)
        log(f"   temperature ceiling {rec['temperature_ceiling'].get('temperature_ceiling_c')} °C; inline connectors tolerated: {rec['connector_budget']['max_inline_connectors']}")
        mc = monte_carlo(link, h.tolerances, n=n_mc, seed=0, phy=h.phy)
        rec["monte_carlo"] = mc.to_dict()
        log(f"   Monte Carlo n={mc.n}: P(pass) {mc.p_pass:.2f}, margin p5 {mc.margin_percentiles['p5']:+.2f} dB, eye p5 {mc.eye_percentiles['p5'] * 1e3:.0f} mV")
        record["figures"].append(fig_montecarlo(mc, figs / f"{key}_mc.png", f"{h.name}: Monte Carlo ({mc.n} draws)"))
        sens = sensitivity(link, h.tolerances, phy=h.phy)
        rec["sensitivity"] = sens
        log("   sensitivity: " + ", ".join(f"{s['parameter']} {s['margin_swing_db']:.2f} dB" for s in sens[:4]))
        record["figures"].append(fig_tornado(sens, figs / f"{key}_tornado.png", f"{h.name}: sensitivity"))
        record["harnesses"][key] = rec

    log("== validation against the synthesiser")
    for sk in (True, False):
        v = validate_scaling(lengths=(3.0, 10.0, 15.0, 25.0), temperatures=(23.0, 85.0, 125.0), skin_scaling=sk)
        record["validation"][f"scaling_{'physical' if sk else 'stack'}"] = {k: v[k] for k in ("l0_m", "noise_db", "skin_scaling", "rows", "model")} | \
            {"extrapolation": {k: v["extrapolation"][k] for k in ("max_err_db", "err_at_2_5ghz_db")}}
        worst = max(v["rows"], key=lambda r: r["il_max_err_db"])
        log(f"   scaling ({'physical' if sk else 'stack'} law): worst IL error {worst['il_max_err_db']:.3f} dB at {worst['length_m']} m / {worst['temperature_c']} °C; extrapolation to 2.5 GHz {v['extrapolation']['err_at_2_5ghz_db']:+.3f} dB")
        record["figures"].append(fig_scaling(v, figs / f"validation_scaling_{'physical' if sk else 'stack'}.png"))
        if sk:
            record["figures"].append(fig_extrapolation(v, figs / "validation_extrapolation.png"))
    vc = validate_cascade()
    record["validation"]["cascade"] = {"same_verdict": vc["same_verdict"], "headline_diff_db": vc["headline_diff_db"], "per_quantity": vc["per_quantity"],
                                       "truth_verdict": vc["truth"]["verdict"], "twin_verdict": vc["twin"]["verdict"]}
    log(f"   cascade: same verdict {vc['same_verdict']} ({vc['twin']['verdict']}), headline diff {vc['headline_diff_db']:+.3f} dB")
    ve = validate_eye(n_symbols=3000 if quick else 6000)
    record["validation"]["eye"] = ve
    for r in ve:
        log(f"   eye {r['length_m']} m: PDA raw {r['pda_raw_v'] * 1e3:.0f} / sim {r['sim_raw_v'] * 1e3:.0f} mV, equalised PDA {r['pda_eq_v'] * 1e3:.0f} / sim {r['sim_eq_v'] * 1e3:.0f} mV, bounds hold {r['bound_holds_raw'] and r['bound_holds_eq']}")
    record["seconds"] = time.time() - t0
    (out / "demo_record.json").write_text(json.dumps(record, indent=1, default=_json_default), encoding="utf-8")
    _write_index(out, record)
    log(f"done in {record['seconds']:.0f} s -> {out / 'index.html'}")
    return record


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _write_index(out: Path, record: dict) -> None:
    rows = []
    for key, rec in record["harnesses"].items():
        ev, eye, mc = rec["evaluation"], rec["eye"], rec["monte_carlo"]
        rows.append(f"<h2>{key}: {rec['describe']['name']}</h2>"
                    f"<p><b>{ev['verdict']}</b> - headline {ev['headline']} margin {ev['headline_margin_db']:+.2f} dB at {ev['temperature_c']:.0f} °C; "
                    f"{ev['cable_length_m']:.1f} m of cable, {ev['n_connectors']} connectors. Eye ({eye['phy']}): raw {eye['height_raw_v'] * 1e3:.0f} mV, "
                    f"equalised {eye['height_eq_v'] * 1e3:.0f} mV, with noise {eye['height_noise_v'] * 1e3:.0f} mV. "
                    f"Monte Carlo P(pass) = {mc['p_pass']:.2f}, margin p5 {mc['margin_percentiles_db']['p5']:+.2f} dB.</p>"
                    + "".join(f'<img src="figures/{key}_{k}.png">' for k in ("link", "elements", "eye", "maxlength", "mc", "tornado")))
    val = "".join(f'<img src="figures/{Path(p).name}">' for p in record["figures"] if "validation" in p)
    html = ("<!doctype html><meta charset='utf-8'><title>linktwin demo</title><style>body{font-family:sans-serif;max-width:1100px;margin:auto}img{max-width:100%;margin:4px 0}</style>"
            f"<h1>linktwin {record['linktwin']} - demonstration</h1>" + "".join(rows) + "<h2>validation against the synthesiser</h2>" + val)
    (out / "index.html").write_text(html, encoding="utf-8")
