"""
Uncertainty propagation through the twin: Monte Carlo over what is not known exactly.

Each Monte-Carlo draw perturbs the link's parameters from their stated distributions and re-evaluates
the link-segment margins and (optionally) the eye:

    cable       a, b (loss coefficients) with the relative standard uncertainties of the fit or of the
                platform's budget (project F: u_cal, u_repeat ...) or of project E's prediction interval;
                Z_d with its uncertainty; length with a cutting tolerance; temperature with the label
                uncertainty
    connectors  z_diff, pin inductance, pad capacitance, contact asymmetry with tolerances
    PHY         return-loss magnitude at its limit with a uniformly random phase (the worst case is not
                a single phase, it is whichever one lines up with the link's own reflections)

Outputs: the distribution of the headline margin, P(pass), per-quantity pass probabilities, the eye
height distribution, and a one-at-a-time sensitivity table (tornado): the change of the headline
margin and of the equalised eye height when each parameter moves by +-1 sigma with the rest nominal.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .elements import CableSegment, Connector
from .eye import PhySpec, eye_analysis
from .link import Link

__all__ = ["Tolerances", "monte_carlo", "sensitivity", "MCResult"]


@dataclass
class Tolerances:
    """Relative (fraction) or absolute standard uncertainties, 1 sigma."""
    cable_a_rel: float = 0.02
    cable_b_rel: float = 0.05
    cable_z_ohm: float = 1.0
    cable_length_rel: float = 0.005
    temperature_k: float = 2.0
    conn_z_ohm: float = 5.0
    conn_l_pin_rel: float = 0.3
    conn_c_pad_rel: float = 0.3
    conn_asym_c_abs: float = 0.01
    phy_rl_db: float = 20.0          # PHY return loss magnitude (fixed), phase random

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _perturb(link: Link, tol: Tolerances, rng: np.random.Generator, which: str | None = None, sigma: float | None = None) -> Link:
    """A deep copy of the link with parameters perturbed (all, or one named parameter by ``sigma`` standard deviations)."""
    lk = copy.deepcopy(link)

    def draw(scale):
        return rng.normal(0, 1) * scale if which is None else (sigma * scale)
    for e in lk.elements:
        if isinstance(e, CableSegment):
            m = e.model
            if which in (None, "cable_a"):
                fa = 1 + draw(tol.cable_a_rel); m.a_d *= fa; m.a_c *= fa
            if which in (None, "cable_b"):
                fb = 1 + draw(tol.cable_b_rel); m.b_d *= fb; m.b_c *= fb
            if which in (None, "cable_z"):
                dz = draw(tol.cable_z_ohm); m.z_d += dz; m.z_c += dz * 0.35
            if which in (None, "cable_length"):
                e.length_m *= 1 + draw(tol.cable_length_rel)
            if which in (None, "temperature"):
                e.temperature_c += draw(tol.temperature_k)
        elif isinstance(e, Connector):
            if which in (None, "conn_z"):
                dz = draw(tol.conn_z_ohm); e.z_diff += dz; e.z_comm += 0.35 * dz
            if which in (None, "conn_l_pin"):
                e.l_pin *= max(1 + draw(tol.conn_l_pin_rel), 0.05)
            if which in (None, "conn_c_pad"):
                e.c_pad *= max(1 + draw(tol.conn_c_pad_rel), 0.05)
            if which in (None, "conn_asym"):
                e.asym_c += draw(tol.conn_asym_c_abs)
    return lk


PARAMS = ["cable_a", "cable_b", "cable_z", "cable_length", "temperature", "conn_z", "conn_l_pin", "conn_c_pad", "conn_asym"]


@dataclass
class MCResult:
    n: int
    p_pass: float
    margins: np.ndarray
    verdicts: list[str]
    headlines: list[str]
    quantity_pass: dict
    eye_heights: np.ndarray | None
    margin_percentiles: dict
    eye_percentiles: dict | None
    tolerances: dict

    def to_dict(self) -> dict:
        return {"n": self.n, "p_pass": self.p_pass, "margin_percentiles_db": self.margin_percentiles, "quantity_pass_probability": self.quantity_pass,
                "eye_height_percentiles_v": self.eye_percentiles, "headline_counts": {h: self.headlines.count(h) for h in set(self.headlines)},
                "tolerances": self.tolerances}


def monte_carlo(link: Link, tol: Tolerances | None = None, n: int = 200, seed: int = 0, phy: str | PhySpec | None = "1000base-t1",
                gamma_phase_random: bool = True) -> MCResult:
    tol = tol or Tolerances()
    rng = np.random.default_rng(seed)
    margins, verdicts, heads, eyes = [], [], [], []
    qpass: dict[str, int] = {}
    for _ in range(n):
        lk = _perturb(link, tol, rng)
        r = lk.evaluate()
        margins.append(r.headline_margin if r.headline_margin is not None else np.nan)
        verdicts.append(r.verdict)
        heads.append(r.headline or "")
        for row in r.rows:
            if row["kind"] is not None:
                qpass[row["quantity"]] = qpass.get(row["quantity"], 0) + int(bool(row["passed"]))
        if phy is not None:
            g = 10 ** (-tol.phy_rl_db / 20)
            gs = g * np.exp(1j * rng.uniform(0, 2 * np.pi)) if gamma_phase_random else g
            gl = g * np.exp(1j * rng.uniform(0, 2 * np.pi)) if gamma_phase_random else g
            e = eye_analysis(lk, phy, gs, gl, simulate=False)
            eyes.append(e.height_noise_v)
    margins = np.array(margins, float)
    eyes_arr = np.array(eyes) if eyes else None
    pct = lambda a: {f"p{q}": float(np.nanpercentile(a, q)) for q in (5, 50, 95)}
    return MCResult(n, float(np.mean([v == "PASS" for v in verdicts])), margins, verdicts, heads, {k: v / n for k, v in qpass.items()},
                    eyes_arr, pct(margins), pct(eyes_arr) if eyes_arr is not None else None, tol.to_dict())


def sensitivity(link: Link, tol: Tolerances | None = None, phy: str | PhySpec | None = "1000base-t1") -> list[dict]:
    """One-at-a-time +-1 sigma: change of the headline margin and of the equalised eye height per parameter."""
    tol = tol or Tolerances()
    rng = np.random.default_rng(0)
    base = link.evaluate()
    base_eye = eye_analysis(link, phy, simulate=False).height_eq_v if phy else None
    out = []
    for name in PARAMS:
        row = {"parameter": name}
        for sgn, key in ((+1, "plus"), (-1, "minus")):
            lk = _perturb(link, tol, rng, which=name, sigma=sgn)
            r = lk.evaluate()
            row[f"margin_{key}_db"] = (r.headline_margin or 0.0) - (base.headline_margin or 0.0)
            if phy:
                row[f"eye_{key}_v"] = eye_analysis(lk, phy, simulate=False).height_eq_v - base_eye
        row["margin_swing_db"] = abs(row["margin_plus_db"] - row["margin_minus_db"])
        if phy:
            row["eye_swing_v"] = abs(row["eye_plus_v"] - row["eye_minus_v"])
        out.append(row)
    out.sort(key=lambda r: -r["margin_swing_db"])
    return out
