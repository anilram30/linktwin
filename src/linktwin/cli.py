"""
Command line.

    linktwin evaluate HARNESS.toml [--json] [--fig PNG]      the whole link against the link-segment limits
    linktwin eye HARNESS.toml [--phy P] [--fig PNG]          pulse response, PDA eye, equalised eye, simulation
    linktwin maxlength HARNESS.toml [--temperatures ...]     the longest cable that still passes (and eye criterion)
    linktwin limiting HARNESS.toml                           which element limits the link
    linktwin connectors HARNESS.toml [--max-n N]             how many inline connectors the link tolerates
    linktwin montecarlo HARNESS.toml [-n N] [--fig PNG]      P(pass), margin and eye distributions
    linktwin sensitivity HARNESS.toml [--fig PNG]            one-at-a-time tornado
    linktwin validate [--out DIR]                            the twin against the synthesiser
    linktwin demo OUT [--archive-job DIR] [--production-dataset CSV] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .harness import load_harness


def _rows(rows):
    w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) for r in rows)


def cmd_evaluate(a):
    h = load_harness(a.harness)
    r = h.link.evaluate()
    if a.json:
        print(json.dumps({"harness": h.describe(), "evaluation": r.to_dict()}, indent=1, default=str))
    else:
        print(f"{h.name}: {h.link.cable_length_m:.2f} m of cable, {h.link.n_connectors} connectors, {h.temperature_c:.0f} °C, standard {h.link.cable_type.id}")
        for s in h.sources:
            print("  cable source: " + ", ".join(f"{k}={v}" for k, v in s.items() if k not in ("intervals90", "predicted", "design", "u_rel")))
        print(f"verdict {r.verdict}; headline {r.headline} margin {r.headline_margin:+.2f} dB" + (f" at {r.headline_x / 1e6:.1f} MHz" if r.headline_x else ""))
        tab = [["quantity", "pair", "pass", "margin dB", "worst at MHz"]]
        for row in r.rows:
            if row["kind"]:
                tab.append([row["quantity"], row["pair"], "yes" if row["passed"] else "NO", f"{row['worst_margin']:+.2f}" if row["worst_margin"] is not None else "-",
                            f"{row['x_worst'] / 1e6:.1f}" if row.get("x_worst") is not None else "-"])
        print(_rows(tab))
        print("IL: " + ", ".join(f"{k} {v:.2f} dB" for k, v in r.il_db.items()) + f"; RL min {r.rl_min_db:.1f} dB; LCL min {r.lcl_min_db:.1f} dB")
        for w in r.warnings:
            print("warning: " + w)
    if a.fig:
        from .figures import fig_link
        print("figure: " + fig_link(h.link, a.fig))
    return 0 if r.verdict == "PASS" else 1


def cmd_eye(a):
    from .eye import eye_analysis
    h = load_harness(a.harness)
    e = eye_analysis(h.link, a.phy or h.phy, simulate=not a.no_sim, n_symbols=a.symbols)
    d = e.to_dict()
    if a.json:
        print(json.dumps(d, indent=1))
    else:
        print(f"{h.name} with {e.phy} ({e.baud / 1e6:.1f} MBd): main cursor {e.main_cursor:.3f}, pulse bandwidth {e.f_max_hz / 1e9:.1f} GHz (band {e.band_hz / 1e6:.0f} MHz)")
        print(f"raw eye        : height {e.height_raw_v * 1e3:.0f} mV, width {e.width_raw_ui:.2f} UI, closure {e.closure_raw:.2f}, open {e.open_raw}"
              + (f"; simulated {e.sim_height_raw_v * 1e3:.0f} mV" if e.sim_height_raw_v is not None else ""))
        print(f"equalised eye  : height {e.height_eq_v * 1e3:.0f} mV, width {e.width_eq_ui:.2f} UI, closure {e.closure_eq:.2f}, open {e.open_eq}"
              + (f"; simulated {e.sim_height_eq_v * 1e3:.0f} mV" if e.sim_height_eq_v is not None else ""))
        print(f"with noise     : {e.height_noise_v * 1e3:.0f} mV at BER 1e-12")
    if a.fig:
        from .figures import fig_pulse_eye
        print("figure: " + fig_pulse_eye(e, a.fig))
    return 0


def cmd_maxlength(a):
    from .inverse import max_length
    h = load_harness(a.harness)
    temps = [float(t) for t in a.temperatures.split(",")] if a.temperatures else [h.temperature_c]
    curve = []
    for T in temps:
        m = max_length(h.link, T)
        row = {"temperature_c": T, **m}
        if a.min_eye_mv is not None:
            me = max_length(h.link, T, phy=h.phy, min_eye_v=a.min_eye_mv / 1e3)
            row["max_length_eye_m"] = me["max_length_m"]
        curve.append(row)
        print(f"{T:6.1f} °C: max length {m['max_length_m']:.2f} m, limited by {m['limited_by']}" + (f"; with eye >= {a.min_eye_mv:.0f} mV: {row['max_length_eye_m']:.2f} m" if a.min_eye_mv is not None else ""))
    if a.fig:
        from .figures import fig_max_length
        print("figure: " + fig_max_length(curve, a.fig))
    return 0


def cmd_limiting(a):
    from .inverse import limiting_element, temperature_ceiling
    h = load_harness(a.harness)
    tab = [["element", "type", "margin without (dB)", "gain (dB)", "headline without"]]
    for d in limiting_element(h.link):
        tab.append([d["element"], d["type"], f"{d['margin_without_db']:+.2f}", f"{d['gain_db']:+.2f}", d["headline_without"]])
    print(_rows(tab))
    tc = temperature_ceiling(h.link)
    print(f"temperature ceiling: {tc.get('temperature_ceiling_c')} °C ({tc.get('limited_by') or tc.get('note')})")
    return 0


def cmd_connectors(a):
    from .elements import Connector
    from .inverse import connector_budget
    h = load_harness(a.harness)
    r = connector_budget(h.link, Connector(z_diff=a.z_diff, asym_c=a.asym_c, name="inline"), a.max_n, h.temperature_c)
    tab = [["inline connectors", "verdict", "headline", "margin (dB)", "LCL min (dB)"]]
    for row in r["table"]:
        tab.append([row["n_inline"], row["verdict"], row["headline"], f"{row['headline_margin_db']:+.2f}", f"{row['lcl_min_db']:.1f}"])
    print(_rows(tab))
    print(f"tolerated: {r['max_inline_connectors']} inline connectors of {a.z_diff:.0f} ohm / asym {a.asym_c}")
    return 0


def cmd_montecarlo(a):
    from .uncertainty import monte_carlo
    h = load_harness(a.harness)
    mc = monte_carlo(h.link, h.tolerances, n=a.n, seed=a.seed, phy=h.phy)
    d = mc.to_dict()
    if a.json:
        print(json.dumps(d, indent=1))
    else:
        print(f"{h.name}: {mc.n} draws, P(pass) = {mc.p_pass:.3f}")
        print("headline margin p5/p50/p95: " + "/".join(f"{d['margin_percentiles_db'][k]:+.2f}" for k in ("p5", "p50", "p95")) + " dB")
        if d["eye_height_percentiles_v"]:
            print("equalised eye p5/p50/p95 : " + "/".join(f"{d['eye_height_percentiles_v'][k] * 1e3:.0f}" for k in ("p5", "p50", "p95")) + " mV")
        print("pass probability per quantity: " + ", ".join(f"{k} {v:.2f}" for k, v in d["quantity_pass_probability"].items()))
        print("headline counts: " + ", ".join(f"{k} {v}" for k, v in d["headline_counts"].items()))
    if a.fig:
        from .figures import fig_montecarlo
        print("figure: " + fig_montecarlo(mc, a.fig))
    return 0


def cmd_sensitivity(a):
    from .uncertainty import sensitivity
    h = load_harness(a.harness)
    s = sensitivity(h.link, h.tolerances, phy=h.phy)
    tab = [["parameter", "margin +1σ (dB)", "margin −1σ (dB)", "swing (dB)", "eye +1σ (mV)", "eye −1σ (mV)"]]
    for r in s:
        tab.append([r["parameter"], f"{r['margin_plus_db']:+.3f}", f"{r['margin_minus_db']:+.3f}", f"{r['margin_swing_db']:.3f}", f"{r.get('eye_plus_v', 0) * 1e3:+.1f}", f"{r.get('eye_minus_v', 0) * 1e3:+.1f}"])
    print(_rows(tab))
    if a.fig:
        from .figures import fig_tornado
        print("figure: " + fig_tornado(s, a.fig))
    return 0


def cmd_validate(a):
    from .validate import validate_cascade, validate_eye, validate_scaling
    out = Path(a.out) if a.out else None
    rec = {}
    for sk in (True, False):
        v = validate_scaling(skin_scaling=sk)
        law = "physical" if sk else "stack"
        print(f"scaling ({law} temperature law), twin from one {v['l0_m']:.0f} m measurement:")
        for r in v["rows"]:
            print(f"  {r['length_m']:5.1f} m {r['temperature_c']:6.1f} °C  IL max err {r['il_max_err_db']:.3f} dB (at 600 MHz {r['il_err_600_db']:+.3f} of {r['il_600_truth_db']:.2f})  RL min truth/twin {r['rl_min_truth_db']:.1f}/{r['rl_min_twin_db']:.1f} dB  phase {r['phase_max_err_deg']:.1f}°")
        print(f"  extrapolation 0.6-2.5 GHz: max err {v['extrapolation']['max_err_db']:.3f} dB")
        rec[f"scaling_{law}"] = {k: v[k] for k in ("rows", "model")}
        if out:
            from .figures import fig_extrapolation, fig_scaling
            out.mkdir(parents=True, exist_ok=True)
            fig_scaling(v, out / f"validation_scaling_{law}.png")
            if sk:
                fig_extrapolation(v, out / "validation_extrapolation.png")
    vc = validate_cascade()
    print(f"cascade: truth {vc['truth']['verdict']} / twin {vc['twin']['verdict']}, headline diff {vc['headline_diff_db']:+.3f} dB")
    for q, d in vc["per_quantity"].items():
        print(f"  {q:18s} truth {d['truth_margin_db']:+.2f}  twin {d['twin_margin_db']:+.2f}  diff {d['diff_db']:+.3f} dB")
    rec["cascade"] = {k: vc[k] for k in ("per_quantity", "same_verdict", "headline_diff_db")}
    ve = validate_eye(n_symbols=a.symbols)
    for r in ve:
        print(f"eye {r['length_m']:4.1f} m: PDA raw {r['pda_raw_v'] * 1e3:.0f} mV / sim {r['sim_raw_v'] * 1e3:.0f} mV; equalised PDA {r['pda_eq_v'] * 1e3:.0f} / sim {r['sim_eq_v'] * 1e3:.0f} mV; bound holds {r['bound_holds_raw'] and r['bound_holds_eq']}")
    rec["eye"] = ve
    if out:
        (out / "validation.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return 0


def cmd_demo(a):
    from .demo import run_demo
    run_demo(a.out, a.archive_job, a.production_dataset, a.quick)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="linktwin", description="Full-link digital twin: predictive signal integrity of automotive Ethernet harnesses")
    p.add_argument("--version", action="version", version=f"linktwin {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("evaluate", help="the whole link against the link-segment limits"); s.add_argument("harness"); s.add_argument("--json", action="store_true"); s.add_argument("--fig"); s.set_defaults(func=cmd_evaluate)
    s = sub.add_parser("eye", help="pulse response and eye"); s.add_argument("harness"); s.add_argument("--phy"); s.add_argument("--no-sim", action="store_true")
    s.add_argument("--symbols", type=int, default=6000); s.add_argument("--json", action="store_true"); s.add_argument("--fig"); s.set_defaults(func=cmd_eye)
    s = sub.add_parser("maxlength", help="maximum cable length"); s.add_argument("harness"); s.add_argument("--temperatures", help="comma separated, °C")
    s.add_argument("--min-eye-mv", type=float); s.add_argument("--fig"); s.set_defaults(func=cmd_maxlength)
    s = sub.add_parser("limiting", help="which element limits the link"); s.add_argument("harness"); s.set_defaults(func=cmd_limiting)
    s = sub.add_parser("connectors", help="inline connector budget"); s.add_argument("harness"); s.add_argument("--max-n", type=int, default=8)
    s.add_argument("--z-diff", type=float, default=88.0); s.add_argument("--asym-c", type=float, default=0.03); s.set_defaults(func=cmd_connectors)
    s = sub.add_parser("montecarlo", help="Monte Carlo over the tolerances"); s.add_argument("harness"); s.add_argument("-n", type=int, default=200)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--json", action="store_true"); s.add_argument("--fig"); s.set_defaults(func=cmd_montecarlo)
    s = sub.add_parser("sensitivity", help="one-at-a-time tornado"); s.add_argument("harness"); s.add_argument("--fig"); s.set_defaults(func=cmd_sensitivity)
    s = sub.add_parser("validate", help="the twin against the synthesiser"); s.add_argument("--out"); s.add_argument("--symbols", type=int, default=6000); s.set_defaults(func=cmd_validate)
    s = sub.add_parser("demo", help="examples + everything, with figures"); s.add_argument("out"); s.add_argument("--archive-job"); s.add_argument("--production-dataset")
    s.add_argument("--quick", action="store_true"); s.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
