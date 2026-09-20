"""
The eye at the receiver.

Pulse response.  With the link's differential voltage transfer H(f) (matched or mismatched PHYs, see
``Link.transfer``), a transmitter that sends one symbol as a rectangular pulse of one unit interval
shaped by a Gaussian edge filter,

    P_tx(f) = UI sinc(f UI) e^{-j pi f UI} exp(-f^2 / (2 sigma_f^2)),   sigma_f = 1.6832 / (2 pi t_r),

(t_r the 20-80 % rise time), the received pulse is p(t) = F^{-1}{ H(f) P_tx(f) }, evaluated on a
grid dense enough for the eye (``oversample`` points per UI) and long enough for the tail (several
cable delays).  The link's models are evaluated far beyond the measured band: that is what the causal
propagation model of :mod:`mtl` is for, and the band edge is reported with the result.

Peak distortion analysis (PDA).  For PAM-M with level spacing Delta (peak-to-peak differential
V_pp = (M-1) Delta), the sample at the decision instant t_0 + k UI is r = a_0 p_0 + sum_{k != 0} a_k p_k
with a_k in {-(M-1)/2, ..., (M-1)/2} Delta.  The worst-case inner eye opening is

    h_eye(t_0) = Delta [ p_0(t_0) - (M-1) sum_{k != 0} |p_k(t_0)| ],

the eye height is its maximum over t_0, and the eye width is the span of t_0 over which it is
positive.  This is the deterministic worst case, exact for uncorrelated symbols and a bound for
any sequence; a bit-level simulation over random symbols (``simulate``) is provided to validate
it, and always shows a slightly larger opening because the worst sequence is rare.

Equalisation.  A decision-feedback equaliser of N taps removes the first N post-cursors from the
sum exactly (ideal DFE); a feed-forward equaliser of (n_pre, n_post) taps is the least-squares
zero-forcing solution on the symbol-spaced pulse and is applied before the DFE.  Noise: receiver and
alien-crosstalk noise of rms sigma_n reduces the height by 2 Q sigma_n with Q = 7.03 for a BER of 1e-12.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .link import Link

__all__ = ["PhySpec", "PHYS", "pulse_response", "cursor_window", "pda_eye", "equalize", "simulate_eye", "EyeResult", "eye_analysis"]

Q_1E12 = 7.03


@dataclass
class PhySpec:
    name: str
    baud: float                  # symbols per second
    levels: int                  # PAM-M
    vpp: float                   # differential peak-to-peak transmit amplitude, V
    t_rise_s: float              # 20-80 % rise time of the transmitter
    rl_phy_db: float = 20.0      # return loss of the PHY ports (used for the mismatch transfer)
    noise_rms_v: float = 0.005   # receiver + alien noise rms at the slicer, V
    dfe_taps: int = 0
    ffe_pre: int = 0
    ffe_post: int = 0
    oversample: int = 32
    f_band_hz: float = 600e6     # the standard's channel band (reported next to the pulse bandwidth)

    @property
    def ui(self) -> float:
        return 1.0 / self.baud

    @property
    def delta(self) -> float:
        return self.vpp / (self.levels - 1)


PHYS = {
    "100base-t1": PhySpec("100BASE-T1", 66.6667e6, 3, 2.0, 5e-9, 20.0, 0.008, dfe_taps=8, ffe_pre=1, ffe_post=2, oversample=16, f_band_hz=66e6),
    "1000base-t1": PhySpec("1000BASE-T1", 750e6, 3, 1.0, 0.45e-9, 20.0, 0.006, dfe_taps=12, ffe_pre=1, ffe_post=3, oversample=32, f_band_hz=600e6),
    "2.5gbase-t1": PhySpec("2.5GBASE-T1", 1.40625e9, 4, 1.0, 0.25e-9, 20.0, 0.004, dfe_taps=16, ffe_pre=2, ffe_post=4, oversample=32, f_band_hz=1.4e9),
}


def pulse_response(link: Link, phy: PhySpec, t_span_s: float | None = None, gamma_s=0.0, gamma_l=0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """(t, p(t), f_max) for one transmitted symbol; t = 0 at the transmitter."""
    dt = phy.ui / phy.oversample
    if t_span_s is None:
        L = getattr(link, "cable_length_m", 0.0) or 0.0
        t_span_s = max(4 * L / (0.6 * 299_792_458.0) + 60 * phy.ui, 200 * phy.ui)
    n = int(2 ** np.ceil(np.log2(t_span_s / dt)))
    f = np.fft.rfftfreq(n, dt)
    fe = np.maximum(f, 1.0)
    H = link.transfer(fe, gamma_s, gamma_l, causal=True)
    ui = phy.ui
    # Gaussian edge with 20-80 % rise time t_r: the step response is a Gaussian CDF of width sigma_t, whose 20-80 %
    # rise is 2 * 0.8416 sigma_t; in frequency sigma_f = 1 / (2 pi sigma_t) = 1.6832 / (2 pi t_r)
    sigma_f = 1.6832 / (2 * np.pi * phy.t_rise_s)
    # the pulse is launched 4 UI after t = 0 so that its Gaussian leading edge does not wrap to the end of the record
    P = ui * np.sinc(f * ui) * np.exp(-1j * np.pi * f * ui) * np.exp(-0.5 * (f / sigma_f) ** 2) * np.exp(-2j * np.pi * f * 4 * ui)
    p = np.fft.irfft(H * P, n) / dt          # volts per volt of transmitted level
    t = np.arange(n) * dt
    return t, p, float(f[-1])


def sample_cursors(t: np.ndarray, p: np.ndarray, ui: float, t0: float, n_pre: int = 8, n_post: int = 64) -> np.ndarray:
    """Cursor values p_k = p(t0 + k UI), k = -n_pre .. n_post (index n_pre is the main cursor)."""
    k = np.arange(-n_pre, n_post + 1)
    return np.interp(t0 + k * ui, t, p, left=0.0, right=0.0)


def equalize(cursors: np.ndarray, n_pre_c: int, ffe_pre: int = 0, ffe_post: int = 0, dfe_taps: int = 0) -> tuple[np.ndarray, dict]:
    """Apply an FFE (least-squares zero forcing on the cursors) and an ideal DFE; returns the residual cursors
    (post-cursors cancelled by the DFE set to 0) and the tap values."""
    c = cursors.copy()
    taps = {}
    n_ffe = ffe_pre + 1 + ffe_post
    if n_ffe > 1:
        # find w (length n_ffe, main at index ffe_pre) minimising ||conv(c, w) - e||^2 with e = unit at the main cursor
        m = c.size
        A = np.zeros((m + n_ffe - 1, n_ffe))
        for j in range(n_ffe):
            A[j:j + m, j] = c
        target = np.zeros(m + n_ffe - 1)
        target[n_pre_c + ffe_pre] = c[n_pre_c]           # keep the main cursor amplitude
        # let the DFE handle the first post-cursors: drop those rows from the LS problem
        rows = np.ones(m + n_ffe - 1, bool)
        for k in range(1, dfe_taps + 1):
            if n_pre_c + ffe_pre + k < rows.size:
                rows[n_pre_c + ffe_pre + k] = False
        w, *_ = np.linalg.lstsq(A[rows], target[rows], rcond=None)
        c = (A @ w)[ffe_pre:ffe_pre + m]
        taps["ffe"] = w.tolist()
    if dfe_taps > 0:
        d = c[n_pre_c + 1:n_pre_c + 1 + dfe_taps].copy()
        c[n_pre_c + 1:n_pre_c + 1 + dfe_taps] = 0.0
        taps["dfe"] = d.tolist()
    return c, taps


def cursor_window(t: np.ndarray, p: np.ndarray, ui: float, rel: float = 1e-4, n_pre: int = 8, n_max: int = 4000) -> tuple[int, int]:
    """(n_pre, n_post): the post-cursor window reaches the last sample where |p| exceeds ``rel`` of the peak, so that
    reflections between connectors (which arrive two cable delays after the main cursor) are inside it."""
    k_peak = int(np.argmax(p))
    big = np.nonzero(np.abs(p) > rel * p[k_peak])[0]
    k_last = int(big[-1]) if big.size else k_peak
    n_post = int(np.ceil((t[k_last] - t[k_peak]) / ui)) + 2
    return n_pre, int(min(max(n_post, 16), n_max))


def pda_eye(t: np.ndarray, p: np.ndarray, phy: PhySpec, n_pre: int = 8, n_post: int | None = None, equalized: bool = False) -> dict:
    """Peak-distortion eye: height (max over the sampling phase), width, main cursor, ISI sum, closure."""
    ui, M, delta = phy.ui, phy.levels, phy.delta
    if n_post is None:
        n_pre, n_post = cursor_window(t, p, ui, n_pre=n_pre)
    k_peak = int(np.argmax(p))
    phases = np.linspace(t[k_peak] - ui, t[k_peak] + ui, 2 * phy.oversample + 1)
    best = None
    heights = []
    for t0 in phases:
        c = sample_cursors(t, p, ui, t0, n_pre, n_post)
        if equalized:
            c, _ = equalize(c, n_pre, phy.ffe_pre, phy.ffe_post, phy.dfe_taps)
        p0 = c[n_pre]
        isi = float(np.sum(np.abs(c)) - abs(p0))
        h = delta * (p0 - (M - 1) * isi)
        heights.append(h)
        if best is None or h > best[0]:
            best = (h, t0, p0, isi, c)
    heights = np.array(heights)
    open_mask = heights > 0
    width = float(np.sum(open_mask) * (phases[1] - phases[0])) if open_mask.any() else 0.0
    h, t0, p0, isi, c = best
    return {"height_v": float(max(h, 0.0)), "width_ui": min(width / ui, 1.0), "t0_s": float(t0), "main_cursor": float(p0),
            "isi_sum": float(isi), "closure": float(1 - max(h, 0.0) / (delta * p0)) if p0 > 0 else 1.0, "open": bool(h > 0),
            "cursors": c.tolist(), "n_pre": n_pre, "n_post": n_post, "phases_s": phases.tolist(), "heights_v": heights.tolist()}


def simulate_eye(t: np.ndarray, p: np.ndarray, phy: PhySpec, n_symbols: int = 4000, seed: int = 0, t0: float | None = None,
                 equalized: bool = False, n_pre: int = 8, n_post: int | None = None) -> dict:
    """Bit-level validation: random PAM-M symbols through the pulse response, folded into an eye; the observed
    inner eye height at the PDA sampling phase (min over all adjacent-level gaps) and the folded traces."""
    rng = np.random.default_rng(seed)
    M, delta, ui, os_ = phy.levels, phy.delta, phy.ui, phy.oversample
    if n_post is None:
        n_pre, n_post = cursor_window(t, p, ui, n_pre=n_pre)
    dt = ui / os_
    levels = (np.arange(M) - (M - 1) / 2) * delta
    sym = rng.integers(0, M, n_symbols)
    a = levels[sym]
    # waveform: symbol impulses (one per UI) convolved with the sampled pulse
    pk = np.interp(np.arange(0, t[-1], dt), t, p)
    x = np.zeros(n_symbols * os_ + pk.size)
    x[::os_][:n_symbols] = a
    y = np.convolve(x, pk)[:n_symbols * os_] * 1.0
    if equalized:
        # emulate the DFE/FFE by subtracting the cancelled post-cursor ISI with known symbols (ideal decisions)
        c = sample_cursors(t, p, ui, t0 if t0 is not None else t[np.argmax(p)], n_pre, n_post)
        ce, taps = equalize(c, n_pre, phy.ffe_pre, phy.ffe_post, phy.dfe_taps)
        # FFE: filter y with the taps; DFE: subtract sum d_k a_{n-k} at sample instants (only exact at sampling instants)
        if "ffe" in taps:
            w = np.array(taps["ffe"])
            yf = np.zeros_like(y)
            for j, wj in enumerate(w):
                shift = (j - phy.ffe_pre) * os_          # y_f[n] = sum_j w_j y[n - shift_j], zero outside the record
                if shift >= 0:
                    yf[shift:] += wj * y[:y.size - shift]
                else:
                    yf[:shift] += wj * y[-shift:]
            y = yf
    # fold: the k-th symbol's decision instant is at t0 + k UI (t0 relative to the pulse start)
    if t0 is None:
        t0 = t[np.argmax(p)]
    i0 = int(round(t0 / dt))
    if equalized and "dfe" in taps:
        # ideal DFE: subtract sum_k d_k a_{n-k} around each decision instant (exact at the instant itself)
        d = np.array(taps["dfe"])
        fb = np.zeros(n_symbols)
        for k, dk in enumerate(d, 1):
            fb[k:] += dk * a[:-k]
        fb_wave = np.zeros_like(y)
        start = i0 - os_ // 2
        seg = np.repeat(fb, os_)
        lo, hi = max(start, 0), min(start + seg.size, y.size)
        fb_wave[lo:hi] = seg[lo - start:hi - start]
        y = y - fb_wave
    idx = i0 + np.arange(n_symbols) * os_
    # keep decision instants whose whole equaliser span lies inside the record and skip the start-up symbols
    guard = (phy.ffe_pre + phy.ffe_post + 1) * os_
    ok = (idx >= guard) & (idx < y.size - guard) & (np.arange(n_symbols) >= n_pre)
    samples = y[idx[ok]]
    syms = sym[ok]
    # inner eye: for every adjacent level pair, min(upper level samples) - max(lower level samples)
    gaps = []
    for m in range(M - 1):
        lo, hi = samples[syms == m], samples[syms == m + 1]
        if lo.size and hi.size:
            gaps.append(float(hi.min() - lo.max()))
    # folded traces for a picture (two UIs wide)
    n_fold = min(400, n_symbols - 4)
    start = i0 - os_ // 2
    traces = np.array([y[start + k * os_: start + k * os_ + 2 * os_] for k in range(20, 20 + n_fold) if start + k * os_ + 2 * os_ <= y.size and start + k * os_ >= 0])
    return {"height_v": float(min(gaps)) if gaps else 0.0, "gaps_v": gaps, "t0_s": float(t0), "traces": traces, "dt_s": dt,
            "n_symbols": int(n_symbols)}


@dataclass
class EyeResult:
    phy: str
    baud: float
    height_raw_v: float
    width_raw_ui: float
    height_eq_v: float
    width_eq_ui: float
    height_noise_v: float           # equalised height minus 2 Q sigma_n
    closure_raw: float
    closure_eq: float
    main_cursor: float
    isi_raw_v: float
    isi_eq_v: float
    open_raw: bool
    open_eq: bool
    sim_height_raw_v: float | None
    sim_height_eq_v: float | None
    f_max_hz: float
    band_hz: float
    pulse: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "pulse"}


def eye_analysis(link: Link, phy: PhySpec | str = "1000base-t1", gamma_s=0.0, gamma_l=0.0, simulate: bool = True, n_symbols: int = 4000,
                 seed: int = 0) -> EyeResult:
    phy = PHYS[phy] if isinstance(phy, str) else phy
    t, p, fmax = pulse_response(link, phy, gamma_s=gamma_s, gamma_l=gamma_l)
    raw = pda_eye(t, p, phy, equalized=False)
    eq = pda_eye(t, p, phy, equalized=True)
    sim_raw = sim_eq = None
    if simulate:
        sim_raw = simulate_eye(t, p, phy, n_symbols, seed, raw["t0_s"], equalized=False, n_pre=raw["n_pre"], n_post=raw["n_post"])
        sim_eq = simulate_eye(t, p, phy, n_symbols, seed, eq["t0_s"], equalized=True, n_pre=eq["n_pre"], n_post=eq["n_post"])
    h_noise = max(eq["height_v"] - 2 * Q_1E12 * phy.noise_rms_v, 0.0)
    return EyeResult(phy.name, phy.baud, raw["height_v"], raw["width_ui"], eq["height_v"], eq["width_ui"], h_noise, raw["closure"], eq["closure"],
                     raw["main_cursor"], phy.delta * (phy.levels - 1) * raw["isi_sum"], phy.delta * (phy.levels - 1) * eq["isi_sum"],
                     raw["open"], eq["open"], sim_raw["height_v"] if sim_raw else None, sim_eq["height_v"] if sim_eq else None, fmax, phy.f_band_hz,
                     {"t": t, "p": p, "raw": raw, "eq": eq, "sim_raw": sim_raw, "sim_eq": sim_eq})
