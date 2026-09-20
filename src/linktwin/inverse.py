"""
Inverse questions a harness designer asks the twin.

    max_length      the longest total cable length (keeping the harness's segment proportions and its
                    connectors) that still passes the link-segment limits at a given temperature, by
                    bisection on the headline margin; and the same for a minimum equalised eye height
    connector_budget how many inline connectors of a given model the link tolerates at a given length
    limiting_element remove each element in turn (replace by a through) and report the headline-margin
                    gain: what to fix first
    temperature_ceiling the highest temperature at which the link still passes
"""
from __future__ import annotations

import copy

import numpy as np

from .elements import CableSegment, Connector, Element
from .eye import PhySpec, eye_analysis
from .link import Link

__all__ = ["max_length", "connector_budget", "limiting_element", "temperature_ceiling"]


class _Thru(Element):
    name = "thru"

    def sparams(self, f):
        s = np.zeros((np.asarray(f).size, 4, 4), complex)
        s[:, 0, 2] = s[:, 2, 0] = s[:, 1, 3] = s[:, 3, 1] = 1.0
        return s

    def describe(self):
        return {"type": "thru"}


def _with_total_length(link: Link, total_m: float) -> Link:
    lk = copy.deepcopy(link)
    L0 = lk.cable_length_m
    if L0 <= 0:
        raise ValueError("the link has no cable segment")
    lk.scale_cable(total_m / L0)
    return lk


def max_length(link: Link, temperature_c: float | None = None, lo_m: float = 0.5, hi_m: float = 40.0, tol_m: float = 0.05,
               phy: str | PhySpec | None = None, min_eye_v: float = 0.0, step_m: float = 1.0) -> dict:
    """The longest total cable length that passes (and, if ``phy`` is given, keeps an equalised eye height
    >= min_eye_v).  Pass/fail is not monotonic in length - a short link can fail mode conversion (the
    connectors' converted signal is not attenuated) while a longer one passes - so the range is first
    scanned in steps of ``step_m`` and the bisection runs between the last passing and the next failing point."""
    def ok(L):
        lk = _with_total_length(link, L)
        if temperature_c is not None:
            lk.set_temperature(temperature_c)
        r = lk.evaluate()
        good = r.verdict == "PASS"
        eye = None
        if phy is not None and good:
            eye = eye_analysis(lk, phy, simulate=False).height_noise_v
            good = good and eye >= min_eye_v
        return good, r, eye
    grid = np.append(np.arange(lo_m, hi_m, step_m), hi_m)
    results = [(L, *ok(L)) for L in grid]
    passing = [i for i, (_, g, _, _) in enumerate(results) if g]
    if not passing:
        L0, _, r0, e0 = results[0]
        return {"max_length_m": 0.0, "limited_by": r0.headline if r0.verdict != "PASS" else "eye", "temperature_c": temperature_c,
                "note": f"no length between {lo_m} and {hi_m} m passes", "scan": [(float(L), g) for L, g, _, _ in results]}
    i = passing[-1]
    if i == len(results) - 1:
        return {"max_length_m": hi_m, "limited_by": None, "temperature_c": temperature_c, "note": f"passes up to the search limit {hi_m} m"}
    a, b = results[i][0], results[i + 1][0]
    r_b, e_b = results[i + 1][2], results[i + 1][3]
    while b - a > tol_m:
        m = 0.5 * (a + b)
        g, r, e = ok(m)
        if g:
            a = m
        else:
            b, r_b, e_b = m, r, e
    limited = r_b.headline if r_b.verdict != "PASS" else "eye"
    out = {"max_length_m": float(a), "limited_by": limited, "temperature_c": temperature_c, "headline_margin_at_fail_db": r_b.headline_margin,
           "eye_criterion_v": min_eye_v if phy else None, "eye_at_fail_v": e_b, "scan": [(float(L), bool(g)) for L, g, _, _ in results]}
    if passing[0] > 0:
        out["note"] = f"fails below about {results[passing[0]][0]:.1f} m ({results[passing[0] - 1][2].headline})"
    return out


def connector_budget(link: Link, connector: Connector, max_n: int = 8, temperature_c: float | None = None) -> dict:
    """How many inline connectors the link tolerates: the cable (total length, first cable's model) is cut into
    n + 1 equal pieces joined by n copies of ``connector``; the harness's own inline connectors (those between
    its cable pieces) are replaced, its end connectors and PCBs are kept.  The largest n that still passes."""
    cables = [i for i, e in enumerate(link.elements) if isinstance(e, CableSegment)]
    if not cables:
        raise ValueError("no cable segment to cut")
    first, last = cables[0], cables[-1]
    total = link.cable_length_m
    model = link.elements[first].model
    T = link.elements[first].temperature_c if temperature_c is None else temperature_c
    out = []
    best = None
    for n in range(0, max_n + 1):
        seg_len = total / (n + 1)
        middle = []
        for k in range(n + 1):
            middle.append(CableSegment(copy.deepcopy(model), seg_len, T, name=f"cable{k + 1}"))
            if k < n:
                middle.append(copy.deepcopy(connector))
        new = [copy.deepcopy(e) for e in link.elements[:first]] + middle + [copy.deepcopy(e) for e in link.elements[last + 1:]]
        lk = Link(new, link.cable_type, link.name)
        r = lk.evaluate()
        out.append({"n_inline": n, "verdict": r.verdict, "headline": r.headline, "headline_margin_db": r.headline_margin, "lcl_min_db": r.lcl_min_db})
        if r.verdict == "PASS":
            best = n
        elif best is not None and n > best + 1:
            break
    return {"max_inline_connectors": best, "table": out, "cable_length_m": total, "connector": connector.describe()}


def limiting_element(link: Link) -> list[dict]:
    """Headline-margin gain from removing each element (replacing it by a through): the ranking of what limits the link."""
    base = link.evaluate()
    out = []
    for i, e in enumerate(link.elements):
        lk = copy.deepcopy(link)
        lk.elements[i] = _Thru()
        r = lk.evaluate()
        out.append({"element": e.name, "type": type(e).__name__, "margin_without_db": r.headline_margin, "gain_db": (r.headline_margin or 0) - (base.headline_margin or 0),
                    "headline_without": r.headline})
    out.sort(key=lambda d: -d["gain_db"])
    return out


def temperature_ceiling(link: Link, lo_c: float | None = None, hi_c: float = 150.0, tol_k: float = 1.0) -> dict:
    """The highest temperature at which the link still passes, from its own temperature (or ``lo_c``) upwards."""
    lo_c = link.temperature_c if lo_c is None else lo_c

    def ok(T):
        lk = copy.deepcopy(link).set_temperature(T)
        return lk.evaluate()
    r = ok(lo_c)
    if r.verdict != "PASS":
        return {"temperature_ceiling_c": None, "note": f"fails at {lo_c:.0f} C", "limited_by": r.headline}
    r = ok(hi_c)
    if r.verdict == "PASS":
        return {"temperature_ceiling_c": hi_c, "note": f"passes up to {hi_c} C"}
    a, b = lo_c, hi_c
    lim = r.headline
    while b - a > tol_k:
        m = 0.5 * (a + b)
        r = ok(m)
        if r.verdict == "PASS":
            a = m
        else:
            b, lim = m, r.headline
    return {"temperature_ceiling_c": a, "limited_by": lim}
