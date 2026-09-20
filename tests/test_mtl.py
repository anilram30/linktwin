import numpy as np

from linktwin.mtl import (
    abcd_line,
    abcd_to_s2,
    extract_line,
    gamma_model,
    mtl_sparams,
    s2_to_abcd,
    zc_model,
)


def test_line_roundtrip_extracts_gamma_and_zc():
    f = np.linspace(1e6, 600e6, 300)
    g = gamma_model(f, 1.7e-5, 2e-10, 0.68, 0.01)
    zc = zc_model(f, 100.0, 25.0)
    s2 = abcd_to_s2(abcd_line(g, zc, 12.0), 100.0)
    g2, zc2 = extract_line(f, s2, 100.0, 12.0, nvp_guess=0.68)
    assert np.allclose(g2, g, rtol=1e-6, atol=1e-9)
    assert np.allclose(zc2, zc, rtol=1e-6)


def test_abcd_s_roundtrip():
    f = np.linspace(1e6, 600e6, 50)
    abcd = abcd_line(gamma_model(f, 1.7e-5, 2e-10, 0.68), zc_model(f, 95.0, 10.0), 3.0)
    back = s2_to_abcd(abcd_to_s2(abcd, 100.0), 100.0)
    assert np.allclose(back, abcd, rtol=1e-8, atol=1e-10)


def test_causal_model_has_hilbert_pair():
    """beta - 2 pi f / v equals alpha_skin: the (1 + j) sqrt(f) pairing that makes a sqrt(f) loss causal."""
    f = np.linspace(1e6, 600e6, 100)
    a = 1.7e-5
    g = gamma_model(f, a, 0.0, 0.68, 0.0)
    v = 0.68 * 299_792_458.0
    assert np.allclose(g.imag - 2 * np.pi * f / v, a * np.sqrt(f) / 8.686)


def test_mtl_matched_line_is_a_delay():
    """A symmetric lossless MTL segment terminated in its own modal impedances has |S21| = 1 and the right delay."""
    f = np.array([10e6, 11e6])
    z_odd, v = 50.0, 0.68 * 299_792_458.0            # differential 100 ohm -> odd 50 ohm; use z0 = 50 so both modes are matched
    L = np.array([[z_odd / v, 0.0], [0.0, z_odd / v]])
    C = np.array([[1 / (z_odd * v), 0.0], [0.0, 1 / (z_odd * v)]])
    s = mtl_sparams(f, np.zeros((2, 2)), L, np.zeros((2, 2)), C, 2.0, z0=50.0)
    assert np.allclose(np.abs(s[:, 2, 0]), 1.0, atol=1e-9)
    assert np.allclose(np.abs(s[:, 0, 0]), 0.0, atol=1e-9)
    phase = np.unwrap(np.angle(s[:, 2, 0]))
    assert np.isclose(-(phase[1] - phase[0]) / (2 * np.pi * (f[1] - f[0])), 2.0 / v, rtol=1e-6)
