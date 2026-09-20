"""
Validation of the twin against ground truth it did not see.

1. Cable scaling: a pair is "measured" at L0 (project A's synthesiser, with measurement noise), the twin
   extracts its per-metre model and predicts the same pair at other lengths and temperatures; the
   truth is the synthesiser run directly at those lengths/temperatures.  Reported: IL, RL and phase
   errors over the band, and the verdict/headline-margin agreement of whole harnesses.
2. Extrapolation: the twin's model evaluated beyond the measured band against the synthesiser there.
3. Eye: PDA worst-case eye against a bit-level simulation over random symbols (the simulation must
   never be worse than the PDA bound, and should be close).
4. Cascade: a harness assembled from the twin's elements against the same harness assembled from the
   synthesiser's true cable pieces and the same connector models, on the link-segment margins.
"""
from __future__ import annotations

import numpy as np
from cablecheck.mixedmode import PortMap, to_mixed_mode
from cablecheck.network import Network
from cablecheck.synth import PairSpec, make_pair, synthesize_measurement

from .elements import (
    ALPHA_RHO,
    BETA_D,
    GAMMA_Z,
    T_REF,
    CableSegment,
    Connector,
    PcbTrace,
)
from .eye import eye_analysis
from .link import Link

__all__ = ["validate_scaling", "validate_cascade", "validate_eye", "truth_pair"]

PMAP = PortMap.single_pair()


def truth_pair(length_m: float, temperature_c: float = 23.0, f: np.ndarray | None = None, base: PairSpec | None = None,
               skin_scaling: bool = True) -> Network:
    """The synthesiser's pair at a length and temperature.

    The synthesiser computes the skin-effect resistance R_s = sqrt(mu0 rho_Cu / pi) / d from the 20 C copper
    resistivity, so on its own it moves only R_dc and tan delta with temperature.  Physically R_s scales with
    sqrt(rho(T)); ``skin_scaling=True`` (the twin's default law) obtains that from the synthesiser by shrinking
    the conductor diameter by sqrt(1 + alpha dT) (R_s is proportional to 1/d and d enters nowhere else).
    ``skin_scaling=False`` is the stack's own convention (projects A/B/E)."""
    f = np.linspace(1e6, 600e6, 1200) if f is None else f
    b = base or PairSpec()
    dT = temperature_c - T_REF
    d_wire = b.d_wire_m / np.sqrt(1 + ALPHA_RHO * dT) if skin_scaling else b.d_wire_m
    spec = PairSpec(length_m=length_m, z_diff=b.z_diff * (1 - GAMMA_Z * dT), z_comm=b.z_comm * (1 - GAMMA_Z * dT), nvp=b.nvp, nvp_even=b.nvp_even,
                    d_wire_m=d_wire, r_dc_ohm_per_m=b.r_dc_ohm_per_m * (1 + ALPHA_RHO * dT), tan_delta=b.tan_delta * (1 + BETA_D * dT),
                    proximity=b.proximity, ripple_amp=b.ripple_amp, ripple_period_m=b.ripple_period_m, roughness=b.roughness, n_segments=b.n_segments, seed=b.seed)
    return make_pair(spec, f, name=f"truth {length_m} m {temperature_c} C")


def _il_rl_ph(net: Network):
    mm = to_mixed_mode(net, PMAP)
    n, fa = ("d", "A", "near"), ("d", "A", "far")
    s21, s11 = mm.param(fa, n), mm.param(n, n)
    return -20 * np.log10(np.abs(s21) + 1e-30), -20 * np.log10(np.abs(s11) + 1e-30), np.unwrap(np.angle(s21))


def validate_scaling(l0_m: float = 10.0, lengths=(3.0, 10.0, 15.0, 25.0), temperatures=(23.0, 85.0, 125.0), noise_db: float = -85.0,
                     base: PairSpec | None = None, f: np.ndarray | None = None, skin_scaling: bool = True) -> dict:
    """Twin from one noisy measurement at l0 vs the synthesiser at other lengths/temperatures.  ``skin_scaling``
    selects the temperature law of both the truth and the twin (True: physical sqrt(rho) skin resistance;
    False: the stack's fixed-R_s convention) - the twin must match either truth when told which it faces."""
    f = np.linspace(1e6, 600e6, 1200) if f is None else f
    base = base or PairSpec(ripple_amp=0.004, roughness=0.003, seed=3)
    truth0 = truth_pair(l0_m, 23.0, f, base, skin_scaling)
    meas = synthesize_measurement(truth0, None, noise_db=noise_db, seed=1, name="measured")
    rows = []
    for L in lengths:
        for T in temperatures:
            seg = CableSegment.from_network(meas, l0_m, PMAP, L, T, 23.0, skin_temperature=skin_scaling)
            tw = seg.network(f)
            tr = truth_pair(L, T, f, base, skin_scaling)
            il_t, rl_t, ph_t = _il_rl_ph(tr)
            il_p, rl_p, ph_p = _il_rl_ph(tw)
            m = f >= 5e6
            rows.append({"length_m": L, "temperature_c": T, "il_max_err_db": float(np.max(np.abs(il_t - il_p)[m])), "il_err_600_db": float(il_p[-1] - il_t[-1]),
                         "il_600_truth_db": float(il_t[-1]), "rl_min_truth_db": float(rl_t.min()), "rl_min_twin_db": float(rl_p.min()),
                         "rl_rms_err_db": float(np.sqrt(np.mean((np.clip(rl_t, 0, 60) - np.clip(rl_p, 0, 60))[m] ** 2))),
                         "phase_max_err_deg": float(np.degrees(np.max(np.abs((ph_t - ph_p)[m]))))})
    # extrapolation: model only, beyond the band
    f2 = np.linspace(600e6, 2.5e9, 400)
    seg = CableSegment.from_network(meas, l0_m, PMAP, 15.0, 23.0, 23.0)
    seg.model.use_raw = False
    il_t, _, _ = _il_rl_ph(truth_pair(15.0, 23.0, f2, base))
    il_p, _, _ = _il_rl_ph(seg.network(f2))
    extrap = {"f_hz": f2.tolist(), "il_truth_db": il_t.tolist(), "il_twin_db": il_p.tolist(), "max_err_db": float(np.max(np.abs(il_t - il_p))),
              "err_at_2_5ghz_db": float(il_p[-1] - il_t[-1])}
    return {"l0_m": l0_m, "noise_db": noise_db, "skin_scaling": skin_scaling, "rows": rows, "extrapolation": extrap, "model": seg.model.to_dict()}


def validate_cascade(l0_m: float = 10.0, split=(6.0, 9.0), temperature_c: float = 23.0, base: PairSpec | None = None) -> dict:
    """Twin harness (cable from a 10 m measurement) vs truth harness (synthesiser pieces at the real lengths)."""
    f = np.linspace(1e6, 600e6, 1200)
    base = base or PairSpec(ripple_amp=0.004, roughness=0.003, seed=3)
    meas = synthesize_measurement(truth_pair(l0_m, 23.0, f, base), None, noise_db=-85.0, seed=1)
    conns = [Connector(name="header"), Connector(name="inline", z_diff=88.0, asym_c=0.02), Connector(name="header2")]
    pcbs = [PcbTrace(length_m=0.04, name="pcb-tx"), PcbTrace(length_m=0.04, name="pcb-rx")]

    class _Fixed(CableSegment):
        """A truth pair used as an element (its S-parameters as they are, no model)."""
        def __init__(self, net, L, T, name):
            self._net, self.length_m, self.temperature_c, self.name = net, L, T, name
            self.model = None
        def sparams(self, f_):
            return self._net.interpolate(np.asarray(f_, float)).s if not np.array_equal(self._net.f, f_) else self._net.s
        def describe(self):
            return {"type": "truth", "name": self.name, "length_m": self.length_m}
    truth_elems = [pcbs[0], conns[0], _Fixed(truth_pair(split[0], temperature_c, f, base), split[0], temperature_c, "c1"), conns[1],
                   _Fixed(truth_pair(split[1], temperature_c, f, base), split[1], temperature_c, "c2"), conns[2], pcbs[1]]
    twin_elems = [pcbs[0], conns[0], CableSegment.from_network(meas, l0_m, PMAP, split[0], temperature_c, 23.0, name="c1"), conns[1],
                  CableSegment.from_network(meas, l0_m, PMAP, split[1], temperature_c, 23.0, name="c2"), conns[2], pcbs[1]]
    rt = Link(truth_elems).evaluate(f)
    rw = Link(twin_elems).evaluate(f)
    per_q = {}
    for a in rt.rows:
        b = next((x for x in rw.rows if x["quantity"] == a["quantity"] and x["pair"] == a["pair"]), None)
        if a["kind"] and b:
            per_q[a["quantity"]] = {"truth_margin_db": a["worst_margin"], "twin_margin_db": b["worst_margin"],
                                    "diff_db": (b["worst_margin"] or 0) - (a["worst_margin"] or 0), "truth_pass": a["passed"], "twin_pass": b["passed"]}
    return {"truth": rt.to_dict(), "twin": rw.to_dict(), "per_quantity": per_q, "same_verdict": rt.verdict == rw.verdict,
            "headline_diff_db": (rw.headline_margin or 0) - (rt.headline_margin or 0)}


def validate_eye(lengths=(5.0, 10.0, 15.0), phy: str = "1000base-t1", n_symbols: int = 6000) -> list[dict]:
    f = np.linspace(1e6, 600e6, 600)
    meas = synthesize_measurement(truth_pair(10.0, 23.0, f), None, noise_db=-85.0, seed=1)
    out = []
    for L in lengths:
        link = Link([Connector(name="header"), CableSegment.from_network(meas, 10.0, PMAP, L, 23.0, 23.0), Connector(name="header2")])
        e = eye_analysis(link, phy, simulate=True, n_symbols=n_symbols)
        out.append({"length_m": L, "pda_raw_v": e.height_raw_v, "sim_raw_v": e.sim_height_raw_v, "pda_eq_v": e.height_eq_v, "sim_eq_v": e.sim_height_eq_v,
                    "width_raw_ui": e.width_raw_ui, "main_cursor": e.main_cursor, "bound_holds_raw": (e.sim_height_raw_v or 0) >= e.height_raw_v - 0.02,
                    "bound_holds_eq": (e.sim_height_eq_v or 0) >= e.height_eq_v - 0.05})
    return out
