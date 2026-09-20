"""
The link: a list of elements between the two PHYs, cascaded into one 4-port, judged against the
link-segment limits of project A, and handed to the eye engine.

    Tx PHY ] pcb ] connector ] cable ] inline connector ] cable ] connector ] pcb [ Rx PHY

Cascading uses cablecheck's block wave-chain matrices on the single-ended 4-ports (near pair = left
group, far pair = right group), so mode conversion and common-mode propagation are carried through
every element, not just the differential path.  The link-segment verdict re-uses project A entirely:
the cascaded network on the standard's grid goes through ``compute_quantities`` and ``evaluate``
with the same limit files a measured harness would face.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from cablecheck.evaluate import Evaluation, evaluate
from cablecheck.limits.library import CableType, load_cable_type
from cablecheck.mixedmode import MixedModeNetwork, PortMap, to_mixed_mode
from cablecheck.network import Network, cascade
from cablecheck.quantities import compute_quantities

from .elements import CableSegment, Element

__all__ = ["Link", "LinkResult"]

PMAP = PortMap.single_pair()


@dataclass
class LinkResult:
    verdict: str
    headline: str | None
    headline_margin: float | None
    headline_x: float | None
    rows: list[dict]
    warnings: list[str]
    length_m: float
    n_connectors: int
    temperature_c: float
    il_db: dict = field(default_factory=dict)      # at the comparison frequencies
    rl_min_db: float | None = None
    lcl_min_db: float | None = None

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "headline": self.headline, "headline_margin_db": self.headline_margin, "headline_x": self.headline_x,
                "results": self.rows, "warnings": self.warnings, "cable_length_m": self.length_m, "n_connectors": self.n_connectors,
                "temperature_c": self.temperature_c, "il_db": self.il_db, "rl_min_db": self.rl_min_db, "lcl_min_db": self.lcl_min_db}


class Link:
    def __init__(self, elements: list[Element], cable_type: str | CableType = "1000base-t1-link-segment", name: str = "link",
                 extra_limit_dirs: list[str] | None = None):
        self.elements = list(elements)
        self.cable_type = cable_type if isinstance(cable_type, CableType) else load_cable_type(cable_type, extra_limit_dirs)
        self.name = name

    # ---- composition
    @property
    def cable_length_m(self) -> float:
        return float(sum(e.length_m for e in self.elements if isinstance(e, CableSegment)))

    @property
    def n_connectors(self) -> int:
        from .elements import Connector
        return sum(1 for e in self.elements if isinstance(e, Connector))

    @property
    def temperature_c(self) -> float:
        ts = [e.temperature_c for e in self.elements if isinstance(e, CableSegment)]
        return float(np.mean(ts)) if ts else 23.0

    def set_temperature(self, t_c: float) -> "Link":
        for e in self.elements:
            if isinstance(e, CableSegment):
                e.temperature_c = float(t_c)
        return self

    def scale_cable(self, factor: float) -> "Link":
        for e in self.elements:
            if isinstance(e, CableSegment):
                e.length_m *= factor
        return self

    def describe(self) -> list[dict]:
        return [e.describe() for e in self.elements]

    # ---- S-parameters
    def sparams(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, float)
        mats = [e.sparams(f) for e in self.elements]
        return cascade(*mats) if len(mats) > 1 else mats[0]

    def network(self, f: np.ndarray) -> Network:
        return Network(np.asarray(f, float), self.sparams(f), np.full(4, 50.0), name=self.name)

    def mixed(self, f: np.ndarray) -> MixedModeNetwork:
        return to_mixed_mode(self.network(f), PMAP)

    def transfer(self, f: np.ndarray, gamma_s: complex | np.ndarray = 0.0, gamma_l: complex | np.ndarray = 0.0, causal: bool = False) -> np.ndarray:
        """Differential voltage transfer 2 V_load / E_source with PHY reflection coefficients Gamma_S, Gamma_L
        (in the 100 ohm reference); equals Sdd21 for matched PHYs:
            H = S21 (1 + G_L)(1 - G_S) / [ (1 - S11 G_S)(1 - S22 G_L) - S12 S21 G_S G_L ]
        ``causal=True`` evaluates the cable segments without the measured gamma residual (the causal fitted
        propagation plus the measured impedance structure), which is what a time-domain response needs."""
        cables = [e for e in self.elements if isinstance(e, CableSegment) and getattr(e.model, "use_raw_gamma", False)]
        try:
            if causal:
                for e in cables:
                    e.model.use_raw_gamma = False
            mm = self.mixed(f)
        finally:
            for e in cables:
                e.model.use_raw_gamma = True
        n, fa = ("d", "A", "near"), ("d", "A", "far")
        s11, s12, s21, s22 = mm.param(n, n), mm.param(n, fa), mm.param(fa, n), mm.param(fa, fa)
        return s21 * (1 + gamma_l) * (1 - gamma_s) / ((1 - s11 * gamma_s) * (1 - s22 * gamma_l) - s12 * s21 * gamma_s * gamma_l)

    # ---- verdict
    def evaluate(self, f: np.ndarray | None = None, nvp: float | None = None, quantities: list[str] | None = None) -> LinkResult:
        """Judge the whole link against the *link-segment* quantities of the cable type (insertion loss, return
        loss, mode conversion, delay, crosstalk); the impedance-profile quantities of a bare cable are not
        link-segment requirements and are left out unless asked for."""
        ct = self.cable_type
        want = quantities or [l.quantity for l in ct.limits if not l.scalar] + ["group_delay", "delay_per_metre", "nvp"]
        if f is None:
            spans = [(seg.fmin_mhz, seg.fmax_mhz) for lim in ct.limits if not lim.scalar for seg in lim.segments]
            fmax = max((hi for _, hi in spans), default=600.0) * 1e6
            fmin = max(min((lo for lo, _ in spans), default=1.0), 0.5) * 1e6
            f = np.linspace(fmin, fmax, 1200)
        net = self.network(f)
        traces = compute_quantities(net, PMAP, length_m=self.cable_length_m or None, t_rise=ct.rise_time_ps * 1e-12, nvp=nvp,
                                    connector_mask_m=ct.connector_mask_m, impedance_window_m=ct.impedance_window_m, want=want)
        ev: Evaluation = evaluate(traces, ct, self.cable_length_m or None)
        hl = ev.headline
        mm = to_mixed_mode(net, PMAP)
        n, fa = ("d", "A", "near"), ("d", "A", "far")
        il = -20 * np.log10(np.abs(mm.param(fa, n)) + 1e-30)
        rl = -20 * np.log10(np.abs(mm.param(n, n)) + 1e-30)
        lcl = -20 * np.log10(np.abs(mm.param(("c", "A", "near"), n)) + 1e-30)
        il_at = {f"{fx / 1e6:.0f}MHz": float(np.interp(fx, f, il)) for fx in (10e6, 100e6, 300e6, 600e6) if f[0] <= fx <= f[-1]}
        return LinkResult(ev.verdict, f"{hl.quantity}[{hl.pair}]" if hl else None, float(hl.worst_margin) if hl and hl.worst_margin is not None else None,
                          float(hl.x_worst) if hl and hl.x_worst is not None else None, ev.summary_rows(), ev.warnings + [],
                          self.cable_length_m, self.n_connectors, self.temperature_c, il_at, float(rl.min()), float(lcl.min()))
