"""
Figures of a link twin (matplotlib, Agg): the link against its limits, the pulse response and eye,
the Monte-Carlo distributions, the sensitivity tornado, the inverse curves and the validation plots.
Every function writes one PNG and returns its path.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cablecheck.mixedmode import PortMap, to_mixed_mode

from .link import Link

__all__ = ["fig_link", "fig_elements", "fig_pulse_eye", "fig_montecarlo", "fig_tornado", "fig_max_length", "fig_scaling", "fig_extrapolation"]

PMAP = PortMap.single_pair()
C_TWIN, C_TRUTH, C_LIMIT, C_ALT, C_GREY = "#1f5fa8", "#d1495b", "#333333", "#2a9d8f", "#9a9a9a"


def _mm_db(net):
    mm = to_mixed_mode(net, PMAP)
    n, fa = ("d", "A", "near"), ("d", "A", "far")
    q = {"IL": -20 * np.log10(np.abs(mm.param(fa, n)) + 1e-30), "RL": -20 * np.log10(np.abs(mm.param(n, n)) + 1e-30),
         "LCL": -20 * np.log10(np.abs(mm.param(("c", "A", "near"), n)) + 1e-30), "LCTL": -20 * np.log10(np.abs(mm.param(("c", "A", "far"), n)) + 1e-30)}
    return q


def _limit_curve(link: Link, quantity: str, f: np.ndarray):
    for lim in link.cable_type.limits:
        if lim.quantity == quantity and not lim.scalar:
            try:
                v, m = lim.evaluate(f, link.cable_length_m or None)
            except Exception:
                return None
            if lim.kind == "range":
                return None
            v = np.where(m, v, np.nan)
            return v
    return None


def fig_link(link: Link, path: str | Path, f: np.ndarray | None = None, truth=None, title: str | None = None) -> str:
    """IL, RL, LCL and LCTL of the whole link against the link-segment limits (and a truth network if given)."""
    f = np.linspace(1e6, 600e6, 1200) if f is None else f
    q = _mm_db(link.network(f))
    qt = _mm_db(truth) if truth is not None else None
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for ax, (key, lim_name) in zip(axs.flat, [("IL", "insertion_loss"), ("RL", "return_loss"), ("LCL", "lcl"), ("LCTL", "lctl")]):
        ax.plot(f / 1e6, q[key], color=C_TWIN, lw=1.4, label="twin")
        if qt is not None:
            ax.plot(truth.f / 1e6, qt[key], color=C_TRUTH, lw=1.0, ls="--", label="truth")
        lim = _limit_curve(link, lim_name, f)
        if lim is not None:
            ax.plot(f / 1e6, lim, color=C_LIMIT, lw=1.2, ls=":", label="limit")
        ax.set_title(key, fontsize=10)
        ax.set_ylabel("dB")
        ax.grid(alpha=0.3)
        if key == "IL":
            ax.invert_yaxis()
        if key != "IL":
            ax.set_ylim(0, 70)
    for ax in axs[1]:
        ax.set_xlabel("frequency (MHz)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(title or f"{link.name}: {link.cable_length_m:.1f} m of cable, {link.n_connectors} connectors, {link.temperature_c:.0f} °C", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def fig_elements(link: Link, path: str | Path, f: np.ndarray | None = None) -> str:
    """The insertion loss of every element on its own, stacked as a budget at four frequencies."""
    f = np.linspace(1e6, 600e6, 600) if f is None else f
    fx = [10e6, 100e6, 300e6, 600e6]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    names, budget = [], []
    for e in link.elements:
        il = _mm_db(e.network(f))["IL"]
        a1.plot(f / 1e6, il, lw=1.2, label=e.name)
        names.append(e.name)
        budget.append([float(np.interp(x, f, il)) for x in fx])
    a1.set_xlabel("frequency (MHz)"); a1.set_ylabel("insertion loss (dB)"); a1.grid(alpha=0.3); a1.legend(fontsize=7, ncol=2); a1.invert_yaxis()
    a1.set_title("per element", fontsize=10)
    budget = np.array(budget)
    bottom = np.zeros(len(fx))
    for i, n in enumerate(names):
        a2.bar([f"{x / 1e6:.0f}" for x in fx], budget[i], bottom=bottom, label=n, width=0.6)
        bottom += budget[i]
    a2.set_xlabel("MHz"); a2.set_ylabel("dB"); a2.set_title("loss budget (sum of elements)", fontsize=10); a2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _delta_v(eye) -> float:
    """Level spacing of the PHY (V) from the eye result: V_pp / (M - 1)."""
    from .eye import PHYS
    for p in PHYS.values():
        if p.name == eye.phy:
            return p.delta
    return 0.5


def fig_pulse_eye(eye, path: str | Path, title: str | None = None) -> str:
    """Pulse response with the sampled cursors, the PDA eye height vs sampling phase, and the simulated eye
    (raw and equalised) with the PDA inner eye drawn on top."""
    P = eye.pulse
    t, p = P["t"], P["p"]
    ui = 1 / eye.baud
    raw, eq, sr, se = P["raw"], P["eq"], P.get("sim_raw"), P.get("sim_eq")
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.6))
    ax = axs[0, 0]
    i_pk = int(np.argmax(p))
    w = (t > t[i_pk] - 6 * ui) & (t < t[i_pk] + 30 * ui)
    ax.plot((t[w] - raw["t0_s"]) / ui, p[w], color=C_TWIN, lw=1.2)
    k = np.arange(-raw["n_pre"], len(raw["cursors"]) - raw["n_pre"])
    ax.stem(k, raw["cursors"], linefmt=C_GREY, markerfmt="o", basefmt=" ")
    ax.set_xlim(-6, 30); ax.set_xlabel("time (UI, 0 = decision instant)"); ax.set_ylabel("p(t) (V/V)"); ax.set_title("pulse response and cursors", fontsize=10); ax.grid(alpha=0.3)
    ax = axs[0, 1]
    ph = (np.array(raw["phases_s"]) - raw["t0_s"]) / ui
    ax.plot(ph, np.array(raw["heights_v"]) * 1e3, color=C_TWIN, label="raw")
    ax.plot((np.array(eq["phases_s"]) - eq["t0_s"]) / ui + (eq["t0_s"] - raw["t0_s"]) / ui, np.array(eq["heights_v"]) * 1e3, color=C_ALT, label="equalised")
    ax.axhline(0, color=C_LIMIT, lw=0.8)
    ax.set_xlabel("sampling phase (UI)"); ax.set_ylabel("PDA eye height (mV)"); ax.set_title("worst-case inner eye vs phase", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for ax, sim, pda, name in ((axs[1, 0], sr, raw, "raw"), (axs[1, 1], se, eq, "equalised")):
        if sim is not None and len(sim["traces"]):
            tr = sim["traces"]
            tt = np.arange(tr.shape[1]) * sim["dt_s"] / ui - 0.5
            ax.plot(tt, tr.T * 1e3, color=C_TWIN, alpha=0.08, lw=0.6)
            ax.set_title(f"{name} eye, simulated {sim['n_symbols']} symbols; PDA height {pda['height_v'] * 1e3:.0f} mV", fontsize=10)
            h = pda["height_v"] * 1e3
            if h > 0:
                # the inner eye sits between adjacent levels: centred on Delta p0 / 2 at both decision instants
                centre = 0.5 * pda["main_cursor"] * 1e3 * _delta_v(eye)
                for x0 in (0.0, 1.0):
                    ax.plot([x0, x0], [centre - h / 2, centre + h / 2], color=C_TRUTH, lw=2.5, label=f"PDA inner eye {h:.0f} mV" if x0 == 0 else None)
                ax.legend(fontsize=8, loc="upper right")
        else:
            ax.text(0.5, 0.5, "no simulation", ha="center", transform=ax.transAxes)
        ax.set_xlabel("time (UI); decision instants at 0 and 1" + (" (DFE exact there)" if name == "equalised" else "")); ax.set_ylabel("mV"); ax.grid(alpha=0.3)
    fig.suptitle(title or f"{eye.phy}: raw eye {eye.height_raw_v * 1e3:.0f} mV, equalised {eye.height_eq_v * 1e3:.0f} mV, with noise {eye.height_noise_v * 1e3:.0f} mV", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def fig_montecarlo(mc, path: str | Path, title: str | None = None) -> str:
    fig, axs = plt.subplots(1, 2 if mc.eye_heights is not None else 1, figsize=(10, 3.6), squeeze=False)
    ax = axs[0, 0]
    m = mc.margins[np.isfinite(mc.margins)]
    ax.hist(m, bins=30, color=C_TWIN, alpha=0.85)
    ax.axvline(0, color=C_TRUTH, lw=1.5)
    ax.set_xlabel("headline margin (dB, negative = fail)"); ax.set_ylabel("draws")
    ax.set_title(f"P(pass) = {mc.p_pass:.3f}, p5/p50/p95 = {mc.margin_percentiles['p5']:.2f}/{mc.margin_percentiles['p50']:.2f}/{mc.margin_percentiles['p95']:.2f} dB", fontsize=9)
    ax.grid(alpha=0.3)
    if mc.eye_heights is not None:
        ax = axs[0, 1]
        ax.hist(mc.eye_heights * 1e3, bins=30, color=C_ALT, alpha=0.85)
        ax.set_xlabel("equalised eye height incl. noise (mV)"); ax.set_ylabel("draws")
        ax.set_title(f"eye p5/p50/p95 = {mc.eye_percentiles['p5'] * 1e3:.0f}/{mc.eye_percentiles['p50'] * 1e3:.0f}/{mc.eye_percentiles['p95'] * 1e3:.0f} mV", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(title or f"Monte Carlo, {mc.n} draws", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def fig_tornado(sens: list[dict], path: str | Path, title: str | None = None) -> str:
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    names = [r["parameter"] for r in sens][::-1]
    y = np.arange(len(names))
    for ax, key, unit, scale in ((axs[0], "margin", "headline margin change (dB)", 1.0), (axs[1], "eye", "equalised eye change (mV)", 1e3)):
        if f"{key}_plus_db" not in sens[0] and f"{key}_plus_v" not in sens[0]:
            ax.axis("off"); continue
        suf = "_db" if key == "margin" else "_v"
        plus = np.array([r[f"{key}_plus{suf}"] for r in sens])[::-1] * scale
        minus = np.array([r[f"{key}_minus{suf}"] for r in sens])[::-1] * scale
        ax.barh(y, plus, color=C_TWIN, label="+1 σ", height=0.6)
        ax.barh(y, minus, color=C_TRUTH, label="−1 σ", height=0.6)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.axvline(0, color=C_LIMIT, lw=0.8); ax.set_xlabel(unit); ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title or "one-at-a-time sensitivity (tornado)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def fig_max_length(curve: list[dict], path: str | Path, title: str | None = None) -> str:
    """max_length results over temperature: one line per criterion present."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    T = [r["temperature_c"] for r in curve]
    ax.plot(T, [r["max_length_m"] for r in curve], "o-", color=C_TWIN, label="link-segment limits")
    if any("max_length_eye_m" in r for r in curve):
        ax.plot(T, [r.get("max_length_eye_m", np.nan) for r in curve], "s--", color=C_ALT, label="limits + eye criterion")
    ax.set_xlabel("temperature (°C)"); ax.set_ylabel("maximum total cable length (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(title or "maximum reach vs temperature", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def fig_scaling(val: dict, path: str | Path) -> str:
    """Validation: IL error at 600 MHz and RL minima, twin vs synthesiser truth, per length and temperature."""
    rows = val["rows"]
    Ts = sorted({r["temperature_c"] for r in rows})
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    for T in Ts:
        rr = [r for r in rows if r["temperature_c"] == T]
        axs[0].plot([r["length_m"] for r in rr], [r["il_max_err_db"] for r in rr], "o-", label=f"{T:.0f} °C")
        axs[1].plot([r["length_m"] for r in rr], [r["il_err_600_db"] for r in rr], "o-", label=f"{T:.0f} °C")
        axs[2].plot([r["length_m"] for r in rr], [r["rl_min_truth_db"] for r in rr], "o-", label=f"truth {T:.0f} °C")
        axs[2].plot([r["length_m"] for r in rr], [r["rl_min_twin_db"] for r in rr], "s--", label=f"twin {T:.0f} °C")
    axs[0].set_ylabel("max |IL error| over 5–600 MHz (dB)"); axs[1].set_ylabel("IL error at 600 MHz (dB)"); axs[2].set_ylabel("minimum return loss (dB)")
    for ax in axs:
        ax.set_xlabel("predicted length (m)"); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7); axs[2].legend(fontsize=6, ncol=2)
    axs[1].axhline(0, color=C_LIMIT, lw=0.8)
    law = "physical sqrt(rho) skin law" if val.get("skin_scaling", True) else "stack convention (fixed R_s)"
    fig.suptitle(f"twin from one {val['l0_m']:.0f} m measurement at 23 °C vs the synthesiser ({law})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def fig_extrapolation(val: dict, path: str | Path) -> str:
    ex = val["extrapolation"]
    f = np.array(ex["f_hz"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.plot(f / 1e9, ex["il_truth_db"], color=C_TRUTH, ls="--", label="synthesiser (truth)")
    a1.plot(f / 1e9, ex["il_twin_db"], color=C_TWIN, label="twin model (fitted to 5–600 MHz)")
    a1.axvline(0.6, color=C_LIMIT, lw=0.8, ls=":")
    a1.set_xlabel("frequency (GHz)"); a1.set_ylabel("insertion loss (dB), 15 m"); a1.invert_yaxis(); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a2.plot(f / 1e9, np.array(ex["il_twin_db"]) - np.array(ex["il_truth_db"]), color=C_TWIN)
    a2.axhline(0, color=C_LIMIT, lw=0.8); a2.set_xlabel("frequency (GHz)"); a2.set_ylabel("twin − truth (dB)"); a2.grid(alpha=0.3)
    a2.set_title(f"max error {ex['max_err_db']:.3f} dB, at 2.5 GHz {ex['err_at_2_5ghz_db']:+.3f} dB", fontsize=9)
    fig.suptitle("extrapolation beyond the measured band with the causal model", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
