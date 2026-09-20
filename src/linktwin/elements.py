"""
Link elements: everything that sits between the transmitter and the receiver, each able to
produce its single-ended 4-port S-matrix [near+, near-, far+, far-] (Z0 = 50 ohm per port) on any
frequency grid.

CableSegment
    A pair described by its differential and common-mode propagation constants and characteristic
    impedances, plus a small mode-conversion term, at any length and temperature.  Built either from a
    *measured* 4-port network of known length (project A/B/F data: gamma and Zc are extracted per
    mode, then fitted to the causal model of :mod:`mtl` so the element can be evaluated outside the
    measured band and at other lengths) or from *coefficients* (a, b, NVP, Z) - the output of project
    E's production-to-performance prediction, i.e. a cable that has not been measured or even made.
    Temperature enters through c0(T) = c0 (1 + alpha_rho dT), a(T) = a sqrt(1 + alpha_rho dT),
    b(T) = b (1 + beta_d dT), Z(T) = Z (1 - gamma_z dT) with the coefficients of projects E/F; the sqrt law of
    the skin-effect term can be switched off (``skin_temperature=False``) to match a model that keeps the
    skin resistance at its 20 C value, as the synthetic instrument stack of projects A/B does.

Connector
    An inline connector or a header: a short two-conductor line of mismatched impedance with contact
    resistance and a slight capacitive asymmetry between the two contacts (the source of mode
    conversion), with lumped pin inductance and pad capacitance at each interface.  Built as a chain
    of ABCD blocks (shunt C, series L, MTL segment, series L, shunt C) and solved to S by cablecheck's
    boundary solver.

PcbTrace
    A differential trace segment on the ECU/PHY side: impedance, length, loss coefficients.

Every element has ``sparams(f)`` and ``describe()``; the link cascades them with cablecheck's block
wave-chain matrices, so a harness is a list.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cablecheck.mixedmode import (
    MixedModeNetwork,
    PortMap,
    from_mixed_mode,
    to_mixed_mode,
)
from cablecheck.network import Network, abcd_to_s
from scipy.optimize import nnls

from .mtl import (
    C0,
    abcd_line,
    abcd_to_s2,
    extract_line,
    gamma_model,
    mtl_sparams,
    zc_model,
)

__all__ = ["Element", "CableSegment", "CableModel", "Connector", "PcbTrace", "ALPHA_RHO", "BETA_D", "GAMMA_Z", "T_REF"]

ALPHA_RHO, BETA_D, GAMMA_Z, T_REF = 0.00393, 0.004, 6e-5, 20.0
PMAP = PortMap.single_pair()


class Element:
    name: str = "element"

    def sparams(self, f: np.ndarray) -> np.ndarray:            # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict:                                # pragma: no cover - interface
        return {"type": type(self).__name__, "name": self.name}

    def network(self, f: np.ndarray) -> Network:
        return Network(np.asarray(f, float), self.sparams(f), np.full(4, 50.0), name=self.name)


@dataclass
class CableModel:
    """Per-metre propagation of a pair: fitted causal models per mode + mode conversion per metre."""
    a_d: float                 # dB/m/sqrt(Hz), differential
    b_d: float                 # dB/m/Hz
    nvp_d: float
    z_d: float                 # ohm, high-frequency differential impedance
    kz_d: float = 0.0          # skin-effect impedance rise coefficient (sqrt(Hz))
    a_c: float = 0.0
    b_c: float = 0.0
    nvp_c: float = 0.66
    z_c: float = 35.0
    kz_c: float = 0.0
    c0_d: float = 0.0          # dB/m, DC-resistance floor (three-term physical basis)
    c0_c: float = 0.0
    conv_per_m: complex = 0.0  # |Sdc21| per metre at f_conv (linear), phase kept
    f_conv: float = 100e6
    source: str = "coefficients"
    fit_rms_db_per_m: float = 0.0
    fit_band_hz: tuple = (0.0, 0.0)
    # hybrid mode: the fitted model is smooth and causal; the extracted gamma and Zc carry the cable's real
    # structure (impedance ripple, roughness) that sets its return loss.  Stored as residuals (extracted -
    # fitted) on the measured grid, so that a later change of the coefficients (temperature, a Monte-Carlo
    # draw) moves the whole curve and the structure rides on top of it.  The gamma residual is measurement
    # noise plus fit residual as much as structure: it makes the frequency-domain twin exact at the measured
    # length but, being white in frequency, spreads over the whole pulse response as non-causal pre-cursors
    # that a worst-case eye analysis would sum up; the eye engine therefore evaluates the link with
    # ``use_raw_gamma`` off (Link.transfer(..., causal=True)) and keeps the impedance structure only.
    raw_f: np.ndarray | None = None
    raw_dgamma_d: np.ndarray | None = None
    raw_dzc_d: np.ndarray | None = None
    raw_dgamma_c: np.ndarray | None = None
    raw_dzc_c: np.ndarray | None = None
    use_raw: bool = True
    use_raw_gamma: bool = True
    # temperature law of the sqrt(f) (skin-effect) conductor term: R_s = sqrt(pi f mu0 rho)/(pi d) scales with
    # sqrt(rho(T)) -> a(T) = a sqrt(1 + alpha dT).  Set False to reproduce a model (such as the synthetic stack
    # of projects A/B, which keeps R_s at the 20 C copper resistivity) in which only R_dc and tan delta move.
    skin_temperature: bool = True

    def temperature_factors(self, temperature_c: float) -> tuple[float, float, float, float]:
        """(f_r, f_a, f_b, f_z): multipliers of c0, a, b and Z at ``temperature_c`` relative to T_REF."""
        dT = temperature_c - T_REF
        fr = 1 + ALPHA_RHO * dT
        fa = float(np.sqrt(max(fr, 1e-3))) if self.skin_temperature else 1.0
        return fr, fa, 1 + BETA_D * dT, 1 - GAMMA_Z * dT

    def gamma_zc(self, f: np.ndarray, mode: str, temperature_c: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
        fr, fa, fb, fz = self.temperature_factors(temperature_c)
        if mode == "d":
            g, z = gamma_model(f, self.a_d * fa, self.b_d * fb, self.nvp_d, self.c0_d * fr), zc_model(f, self.z_d * fz, self.kz_d)
            dg, dz = self.raw_dgamma_d, self.raw_dzc_d
        else:
            g, z = gamma_model(f, self.a_c * fa, self.b_c * fb, self.nvp_c, self.c0_c * fr), zc_model(f, self.z_c * fz, self.kz_c)
            dg, dz = self.raw_dgamma_c, self.raw_dzc_c
        if self.use_raw and self.raw_f is not None and dz is not None:
            # inside the measured band: model + the measured residual structure, tapered to zero over the last
            # 3 % of the band so the extrapolation is continuous
            inside = (f >= self.raw_f[0]) & (f <= self.raw_f[-1])
            w = np.clip((self.raw_f[-1] - f) / (0.03 * (self.raw_f[-1] - self.raw_f[0])), 0, 1)
            z = z + inside * w * (np.interp(f, self.raw_f, dz.real) + 1j * np.interp(f, self.raw_f, dz.imag))
            if self.use_raw_gamma and dg is not None:
                g = g + inside * w * (np.interp(f, self.raw_f, dg.real) + 1j * np.interp(f, self.raw_f, dg.imag))
        return g, z

    def to_dict(self) -> dict:
        return {"hybrid_raw_structure": bool(self.use_raw and self.raw_f is not None), "skin_temperature": self.skin_temperature, "c0_d_db_per_m": self.c0_d, "a_d_db_per_m_sqrtHz": self.a_d, "b_d_db_per_m_Hz": self.b_d, "nvp_d": self.nvp_d, "z_d_ohm": self.z_d, "kz_d": self.kz_d,
                "a_c": self.a_c, "b_c": self.b_c, "nvp_c": self.nvp_c, "z_c_ohm": self.z_c, "conv_per_m_at_f_conv": abs(self.conv_per_m),
                "f_conv_hz": self.f_conv, "source": self.source, "fit_rms_db_per_m": self.fit_rms_db_per_m, "fit_band_hz": list(self.fit_band_hz)}


def _fit_mode(f: np.ndarray, gamma: np.ndarray, zc: np.ndarray, fmin: float = 5e6) -> tuple[float, float, float, float, float, float, float]:
    """(c0, a, b, nvp, z_inf, k_z, rms dB/m) from extracted gamma, Zc over the band above fmin, with the three-term
    physical basis c0 + a sqrt f + b f (project E showed that the two-term basis pushes conductor loss into b, which
    then scales wrongly with temperature)."""
    m = f >= fmin
    ff, g, z = f[m], gamma[m], zc[m]
    alpha_db = g.real * 8.686
    X = np.column_stack([np.ones_like(ff), np.sqrt(ff), ff])
    coef, _ = nnls(X, alpha_db)
    c0, a, b = float(coef[0]), float(coef[1]), float(coef[2])
    rms = float(np.sqrt(np.mean((X @ coef - alpha_db) ** 2)))
    # beta = 2 pi f / v + a sqrt f / 8.686  ->  v from the slope of (beta - a sqrt f / 8.686) vs 2 pi f
    y = g.imag - a * np.sqrt(ff) / 8.686
    inv_v = float(np.sum(y * 2 * np.pi * ff) / np.sum((2 * np.pi * ff) ** 2))
    nvp = float(1 / (inv_v * C0)) if inv_v > 0 else 0.68
    # Zc: real part ~ z_inf (1 + k_z / sqrt f)
    Xz = np.column_stack([np.ones_like(ff), 1 / np.sqrt(ff)])
    cz, *_ = np.linalg.lstsq(Xz, z.real, rcond=None)
    z_inf = float(cz[0])
    k_z = float(max(cz[1] / z_inf, 0.0)) if z_inf > 0 else 0.0
    return c0, a, b, nvp, z_inf, k_z, rms


class CableSegment(Element):
    def __init__(self, model: CableModel, length_m: float, temperature_c: float = 23.0, name: str = "cable"):
        self.model, self.length_m, self.temperature_c, self.name = model, float(length_m), float(temperature_c), name

    # ---- construction
    @classmethod
    def from_network(cls, net: Network, length_m: float, port_map: str | PortMap = PMAP, new_length_m: float | None = None,
                     temperature_c: float = 23.0, measured_at_c: float = 23.0, nvp_guess: float = 0.68, fmin_fit: float = 5e6,
                     name: str = "cable", skin_temperature: bool = True) -> "CableSegment":
        """A measured pair of length ``length_m`` -> a segment of ``new_length_m`` (default: the same)."""
        pm = PortMap.parse(port_map) if isinstance(port_map, str) else port_map
        mm = to_mixed_mode(net.renormalized(50.0), pm)
        pair = pm.pairs[0]
        f = net.f
        model = _model_from_mixed(mm, pair, f, length_m, nvp_guess, fmin_fit)
        model.source = f"measured {net.name or ''} ({length_m} m)".strip()
        # refer the model to T_REF: undo the temperature the measurement was taken at
        model.skin_temperature = skin_temperature
        fr, fa, fb, fz = model.temperature_factors(measured_at_c)
        model.c0_d /= fr; model.c0_c /= fr
        model.a_d /= fa; model.a_c /= fa
        model.b_d /= fb; model.b_c /= fb
        model.z_d /= fz; model.z_c /= fz
        return cls(model, new_length_m if new_length_m is not None else length_m, temperature_c, name)

    @classmethod
    def from_coefficients(cls, a: float, b: float, nvp: float, z_diff: float, length_m: float, z_comm: float = 35.0, nvp_c: float | None = None,
                          temperature_c: float = 23.0, conv_per_m: float = 0.0, name: str = "cable", source: str = "coefficients", c0: float = 0.0) -> "CableSegment":
        """From loss coefficients (dB/m/sqrt(Hz), dB/m/Hz), NVP and impedance - e.g. project E's prediction."""
        model = CableModel(a, b, nvp, z_diff, 0.0, a * 0.9, b * 0.9, nvp_c or nvp - 0.02, z_comm, 0.0, c0, c0, complex(conv_per_m), 100e6, source)
        return cls(model, length_m, temperature_c, name)

    # ---- S-parameters
    def mixed(self, f: np.ndarray) -> MixedModeNetwork:
        f = np.asarray(f, float)
        fe = np.maximum(f, 1.0)
        gd, zd = self.model.gamma_zc(fe, "d", self.temperature_c)
        gc, zc = self.model.gamma_zc(fe, "c", self.temperature_c)
        sdd = abcd_to_s2(abcd_line(gd, zd, self.length_m), 100.0)
        scc = abcd_to_s2(abcd_line(gc, zc, self.length_m), 25.0)
        s = np.zeros((f.size, 4, 4), complex)
        s[:, :2, :2] = sdd
        s[:, 2:, 2:] = scc
        # mode conversion: distributed asymmetry -> |Sdc21| grows ~linearly with length (small), with the
        # differential transmission's phase; reciprocal and symmetric
        conv = self.model.conv_per_m * self.length_m * np.sqrt(fe / self.model.f_conv) * sdd[:, 1, 0] / np.maximum(np.abs(sdd[:, 1, 0]), 1e-12)
        conv = np.where(np.abs(conv) > 0.3, 0.3 * conv / np.maximum(np.abs(conv), 1e-30), conv)
        s[:, 2, 1] = s[:, 1, 2] = s[:, 3, 0] = s[:, 0, 3] = conv * 0.5      # d(near)->c(far) etc.
        labels = [("d", "A", "near"), ("d", "A", "far"), ("c", "A", "near"), ("c", "A", "far")]
        return MixedModeNetwork(f, s, np.array([100.0, 100.0, 25.0, 25.0]), labels, name=self.name)

    def sparams(self, f: np.ndarray) -> np.ndarray:
        return from_mixed_mode(self.mixed(f), PMAP).renormalized(50.0).s

    def describe(self) -> dict:
        return {"type": "CableSegment", "name": self.name, "length_m": self.length_m, "temperature_c": self.temperature_c, "model": self.model.to_dict()}


def _model_from_mixed(mm: MixedModeNetwork, pair: str, f: np.ndarray, length_m: float, nvp_guess: float, fmin_fit: float) -> CableModel:
    def block(mode):
        n, fa = (mode, pair, "near"), (mode, pair, "far")
        return np.stack([np.stack([mm.param(n, n), mm.param(n, fa)], -1), np.stack([mm.param(fa, n), mm.param(fa, fa)], -1)], 1)
    gd, zd = extract_line(f, block("d"), 100.0, length_m, nvp_guess)
    gc, zc = extract_line(f, block("c"), 25.0, length_m, nvp_guess - 0.02)
    c0_d, a_d, b_d, nvp_d, z_d, kz_d, rms = _fit_mode(f, gd, zd, fmin_fit)
    c0_c, a_c, b_c, nvp_c, z_c, kz_c, _ = _fit_mode(f, gc, zc, fmin_fit)
    k = int(np.argmin(np.abs(f - 100e6)))
    conv = mm.param(("c", pair, "far"), ("d", pair, "near"))[k] * 2 / length_m
    f = np.asarray(f, float)
    return CableModel(a_d, b_d, nvp_d, z_d, kz_d, a_c, b_c, nvp_c, z_c, kz_c, c0_d, c0_c, complex(conv), float(f[k]), "measured", rms, (float(f[0]), float(f[-1])),
                      raw_f=f, raw_dgamma_d=gd - gamma_model(f, a_d, b_d, nvp_d, c0_d), raw_dzc_d=zd - zc_model(f, z_d, kz_d),
                      raw_dgamma_c=gc - gamma_model(f, a_c, b_c, nvp_c, c0_c), raw_dzc_c=zc - zc_model(f, z_c, kz_c))


class Connector(Element):
    """Inline connector / header as a mismatched short line with parasitics and contact asymmetry."""

    def __init__(self, z_diff: float = 90.0, length_m: float = 0.025, nvp: float = 0.6, r_contact_ohm: float = 0.005,
                 l_pin_nh: float = 1.0, c_pad_pf: float = 0.3, asym_c: float = 0.01, loss_db_per_m_at_1ghz: float = 2.0,
                 z_comm: float | None = None, name: str = "connector"):
        self.z_diff, self.length_m, self.nvp, self.r_contact, self.l_pin, self.c_pad, self.asym_c = z_diff, length_m, nvp, r_contact_ohm, l_pin_nh * 1e-9, c_pad_pf * 1e-12, asym_c
        self.loss_1ghz, self.z_comm, self.name = loss_db_per_m_at_1ghz, z_comm if z_comm is not None else 0.35 * z_diff, name

    def sparams(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, float)
        fe = np.maximum(f, 1.0)
        v = self.nvp * C0
        zd, zcm = self.z_diff, self.z_comm
        lo, co = zd / 2 / v, 1 / (zd / 2 * v)
        le, ce = 2 * zcm / v, 1 / (2 * zcm * v)
        Ls, M = (lo + le) / 2, (le - lo) / 2
        Cs, Cm = (co + ce) / 2, (co - ce) / 2
        L = np.array([[Ls, M], [M, Ls]])
        C = np.array([[Cs * (1 + self.asym_c), -Cm], [-Cm, Cs * (1 - self.asym_c)]])
        # series resistance: contact + skin-effect loss (dB/m at 1 GHz -> R'(f) = 2 alpha Z)
        r_skin = self.loss_1ghz / 8.686 * 2 * (zd / 2) * np.sqrt(fe / 1e9)      # ohm/m per conductor
        R = np.zeros((f.size, 2, 2))
        R[:, 0, 0] = R[:, 1, 1] = r_skin + self.r_contact / max(self.length_m, 1e-3)
        seg = mtl_sparams(f, R, L, np.zeros((2, 2)), C, self.length_m)
        # lumped parasitics at both interfaces: pad C (shunt, each wire) and pin L (series, each wire)
        w = 2 * np.pi * f
        nf = f.size
        eye = np.eye(2)

        def shunt(y):
            m = np.zeros((nf, 4, 4), complex); m[:, :2, :2] = eye; m[:, 2:, 2:] = eye; m[:, 2, 0] = y; m[:, 3, 1] = y; return m

        def series(z):
            m = np.zeros((nf, 4, 4), complex); m[:, :2, :2] = eye; m[:, 2:, 2:] = eye; m[:, 0, 2] = z; m[:, 1, 3] = z; return m
        from cablecheck.network import cascade
        ends = abcd_to_s(shunt(1j * w * self.c_pad) @ series(1j * w * self.l_pin), np.full(4, 50.0))
        ends_r = abcd_to_s(series(1j * w * self.l_pin) @ shunt(1j * w * self.c_pad), np.full(4, 50.0))
        return cascade(ends, seg, ends_r)

    def describe(self) -> dict:
        return {"type": "Connector", "name": self.name, "z_diff": self.z_diff, "length_m": self.length_m, "nvp": self.nvp, "r_contact_ohm": self.r_contact,
                "l_pin_nH": self.l_pin * 1e9, "c_pad_pF": self.c_pad * 1e12, "asym_c": self.asym_c, "loss_db_per_m_at_1ghz": self.loss_1ghz}


class PcbTrace(Element):
    def __init__(self, z_diff: float = 95.0, length_m: float = 0.05, nvp: float = 0.55, a: float = 3e-6, b: float = 8e-11, name: str = "pcb"):
        self.z_diff, self.length_m, self.nvp, self.a, self.b, self.name = z_diff, length_m, nvp, a, b, name

    def sparams(self, f: np.ndarray) -> np.ndarray:
        seg = CableSegment.from_coefficients(self.a, self.b, self.nvp, self.z_diff, self.length_m, z_comm=0.4 * self.z_diff, name=self.name, source="pcb model")
        return seg.sparams(f)

    def describe(self) -> dict:
        return {"type": "PcbTrace", "name": self.name, "z_diff": self.z_diff, "length_m": self.length_m, "nvp": self.nvp, "a": self.a, "b": self.b}
