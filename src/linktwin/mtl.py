"""
Transmission-line mathematics shared by the elements.

Uniform line of one mode (differential or common) with propagation constant gamma(f) and
characteristic impedance Zc(f), length L: chain (ABCD) matrix

    [V1; I1] = [[cosh(gamma L),       Zc sinh(gamma L)],
                [sinh(gamma L) / Zc,  cosh(gamma L)  ]] [V2; -I2]

and the inverse problem - a measured uniform line of known length L0 gives

    gamma L0 = acosh(A),   Zc = sqrt(B / C),

with the imaginary part of gamma L0 unwrapped against a velocity guess (acosh returns the principal
branch; the true electrical length is many multiples of 2 pi).

Two-conductor MTL (two wires over a reference) with per-unit-length matrices R (2x2), L, G, C:
Z = R + j omega L, Y = G + j omega C; the telegrapher equations d[V; I]/dx = -[[0, Z],[Y, 0]] [V; I] give the
chain matrix of a segment of length l as [V(0); I(0)] = expm(+[[0, Z],[Y, 0]] l) [V(l); I(l)], and the 4-port S follows from cablecheck's boundary solver.  Used for the connector models, which are
short, lossy, mismatched, slightly asymmetric line segments plus lumped parasitics.

Causal propagation model for extrapolation and synthesis (per metre, f in Hz):

    alpha(f) = (c0 + a sqrt(f) + b f) / 8.686   [Np/m]   (c0: DC-resistance floor, project E's physical basis)
    beta(f)  = 2 pi f / v + a sqrt(f) / 8.686   [rad/m]   (the skin-effect phase term that makes
                                                            alpha = k sqrt(omega) causal; the dielectric
                                                            term's phase companion is folded into v)
    Zc(f)    = Z_inf (1 + (1 - j) k_z / sqrt(f))          (skin-effect rise of |Zc| at low frequency)
"""
from __future__ import annotations

import numpy as np
from cablecheck.network import abcd_to_s

__all__ = ["abcd_line", "abcd_to_s2", "s2_to_abcd", "extract_line", "gamma_model", "zc_model", "mtl_sparams", "MU0", "C0"]

MU0 = 4e-7 * np.pi
C0 = 299_792_458.0


def abcd_line(gamma: np.ndarray, zc: np.ndarray, length: float) -> np.ndarray:
    """(nf, 2, 2) chain matrix of a uniform one-mode line."""
    g = gamma * length
    ch, sh = np.cosh(g), np.sinh(g)
    out = np.empty((gamma.size, 2, 2), complex)
    out[:, 0, 0] = ch
    out[:, 0, 1] = zc * sh
    out[:, 1, 0] = sh / zc
    out[:, 1, 1] = ch
    return out


def abcd_to_s2(abcd: np.ndarray, z0: float) -> np.ndarray:
    """2-port chain matrix -> S for a real reference impedance."""
    A, B, C, D = abcd[:, 0, 0], abcd[:, 0, 1], abcd[:, 1, 0], abcd[:, 1, 1]
    den = A + B / z0 + C * z0 + D
    s = np.empty_like(abcd)
    s[:, 0, 0] = (A + B / z0 - C * z0 - D) / den
    s[:, 0, 1] = 2 * (A * D - B * C) / den
    s[:, 1, 0] = 2 / den
    s[:, 1, 1] = (-A + B / z0 - C * z0 + D) / den
    return s


def s2_to_abcd(s: np.ndarray, z0: float) -> np.ndarray:
    S11, S12, S21, S22 = s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1]
    out = np.empty_like(s)
    out[:, 0, 0] = ((1 + S11) * (1 - S22) + S12 * S21) / (2 * S21)
    out[:, 0, 1] = z0 * ((1 + S11) * (1 + S22) - S12 * S21) / (2 * S21)
    out[:, 1, 0] = ((1 - S11) * (1 - S22) - S12 * S21) / (2 * S21) / z0
    out[:, 1, 1] = ((1 - S11) * (1 + S22) + S12 * S21) / (2 * S21)
    return out


def extract_line(f: np.ndarray, s2: np.ndarray, z0: float, length_m: float, nvp_guess: float = 0.68) -> tuple[np.ndarray, np.ndarray]:
    """(gamma per metre, Zc) of a uniform line from its 2-port S (one mode) and its physical length.

    The branch of acosh is chosen by continuity along frequency, seeded by the velocity guess
    2 pi f L / (nvp c0) at the first point."""
    abcd = s2_to_abcd(s2, z0)
    A = 0.5 * (abcd[:, 0, 0] + abcd[:, 1, 1])          # symmetrise (A = D for a uniform line)
    gl = np.arccosh(A.astype(complex))
    gl = np.where(gl.real < 0, -gl, gl)                  # attenuation must be positive
    beta_guess = 2 * np.pi * f * length_m / (nvp_guess * C0)
    # acosh's imaginary part is in [0, pi]; the true value is +-imag + 2 pi k.  The first point takes the candidate
    # nearest the velocity guess; every further point takes the candidate nearest to the previous value plus the
    # guessed increment, so the choice is by continuity along frequency and does not depend on the guess being
    # accurate in absolute terms (the skin-effect delay a sqrt(f)/8.686 is not in the guess).
    im = gl.imag
    beta_l = np.empty(f.size)
    prev = None
    for i in range(f.size):
        target = beta_guess[i] if prev is None else prev + (beta_guess[i] - beta_guess[i - 1])
        cands = np.array([im[i] + 2 * np.pi * np.round((target - im[i]) / (2 * np.pi)), -im[i] + 2 * np.pi * np.round((target + im[i]) / (2 * np.pi))])
        beta_l[i] = cands[np.argmin(np.abs(cands - target))]
        prev = beta_l[i]
    gamma = (gl.real + 1j * beta_l) / length_m
    zc = np.sqrt(abcd[:, 0, 1] / abcd[:, 1, 0])
    zc = np.where(zc.real < 0, -zc, zc)
    return gamma, zc


def gamma_model(f: np.ndarray, a: float, b: float, nvp: float, c0: float = 0.0) -> np.ndarray:
    """alpha = (c0 + a sqrt f + b f)/8.686 with the physical three-term basis of project E (c0: the DC-resistance
    floor of the conductor loss, which scales with rho like the DC resistance, not like sqrt(rho))"""
    f = np.asarray(f, float)
    alpha = (c0 + a * np.sqrt(f) + b * f) / 8.686
    beta = 2 * np.pi * f / (nvp * C0) + a * np.sqrt(f) / 8.686
    return alpha + 1j * beta


def zc_model(f: np.ndarray, z_inf: float, k_z: float) -> np.ndarray:
    f = np.maximum(np.asarray(f, float), 1.0)
    return z_inf * (1 + (1 - 1j) * k_z / np.sqrt(f))


def mtl_sparams(f: np.ndarray, R: np.ndarray, L: np.ndarray, G: np.ndarray, C: np.ndarray, length: float, z0: float = 50.0) -> np.ndarray:
    """4-port S [near1, near2, far1, far2] of a two-conductor MTL segment from per-unit-length matrices.
    R, G may be (2,2) or (nf,2,2); L, C are (2,2)."""
    f = np.asarray(f, float)
    nf = f.size
    R = np.broadcast_to(np.asarray(R, float), (nf, 2, 2)) if np.asarray(R).ndim == 2 else np.asarray(R, float)
    G = np.broadcast_to(np.asarray(G, float), (nf, 2, 2)) if np.asarray(G).ndim == 2 else np.asarray(G, float)
    w = 2 * np.pi * f
    Z = R + 1j * w[:, None, None] * L[None]
    Y = G + 1j * w[:, None, None] * C[None]
    # expm(+[[0, Z],[Y, 0]] l) in closed form: with P = Z Y = V diag(lambda) V^-1 and gamma_i = sqrt(lambda_i),
    #   A = cosh(sqrt(P) l),  B = sqrt(P)^-1 sinh(sqrt(P) l) Z,  C = Y sqrt(P)^-1 sinh(sqrt(P) l),  D = Y A Y^-1
    # (telegrapher: d[V;I]/dx = -[[0,Z],[Y,0]] [V;I], so [V(0);I(0)] = expm(+M l) [V(l);I(l)], cablecheck's convention)
    P = Z @ Y
    lam, V = np.linalg.eig(P)
    gam = np.sqrt(lam.astype(complex))
    Vi = np.linalg.inv(V)
    cosh_ = V @ (np.cosh(gam * length)[:, :, None] * Vi)
    sinc_ = V @ ((np.sinh(gam * length) / gam)[:, :, None] * Vi)
    abcd = np.empty((nf, 4, 4), complex)
    abcd[:, :2, :2] = cosh_
    abcd[:, :2, 2:] = sinc_ @ Z
    abcd[:, 2:, :2] = Y @ sinc_
    abcd[:, 2:, 2:] = Y @ cosh_ @ np.linalg.inv(Y)
    return abcd_to_s(abcd, np.full(4, z0))
