import numpy as np
from cablecheck.mixedmode import PortMap

from linktwin.elements import CableSegment, Connector
from linktwin.eye import PHYS, eye_analysis, pulse_response
from linktwin.inverse import (
    connector_budget,
    limiting_element,
    max_length,
    temperature_ceiling,
)
from linktwin.link import Link
from linktwin.uncertainty import PARAMS, Tolerances, monte_carlo, sensitivity

PMAP = PortMap.single_pair()


def make_link(measured_10m, L=8.0, T=23.0):
    return Link([Connector(name="h1"), CableSegment.from_network(measured_10m, 10.0, PMAP, L, T, 23.0), Connector(name="inline", z_diff=88.0, asym_c=0.02),
                 CableSegment.from_network(measured_10m, 10.0, PMAP, L / 2, T, 23.0, name="c2"), Connector(name="h2")])


def test_pulse_response_is_causal_and_delayed(measured_10m):
    link = make_link(measured_10m)
    phy = PHYS["1000base-t1"]
    t, p, fmax = pulse_response(link, phy)
    k = int(np.argmax(p))
    delay = t[k] - 4 * phy.ui
    assert 12.0 / (0.72 * 3e8) < delay < 12.0 / (0.60 * 3e8) + 2e-9
    assert fmax > 5e9
    # nothing significant before the arrival
    assert np.max(np.abs(p[: int(k * 0.8)])) < 1e-3 * p[k]


def test_pda_bound_holds_against_simulation(measured_10m):
    link = make_link(measured_10m, 6.0)
    e = eye_analysis(link, "1000base-t1", simulate=True, n_symbols=4000)
    assert e.sim_height_raw_v >= e.height_raw_v - 0.005
    assert e.sim_height_eq_v >= e.height_eq_v - 0.01
    assert e.sim_height_eq_v < e.height_eq_v + 0.05      # and it is a tight bound
    assert e.height_eq_v > e.height_raw_v
    assert e.height_noise_v < e.height_eq_v


def test_eye_shrinks_with_length_and_pam_levels(measured_10m):
    e5 = eye_analysis(make_link(measured_10m, 4.0), "1000base-t1", simulate=False)
    e12 = eye_analysis(make_link(measured_10m, 10.0), "1000base-t1", simulate=False)
    assert e12.height_eq_v < e5.height_eq_v
    e25 = eye_analysis(make_link(measured_10m, 4.0), "2.5gbase-t1", simulate=False)
    assert e25.height_eq_v < e5.height_eq_v


def test_inverse_questions(measured_10m):
    link = make_link(measured_10m, 6.0, 85.0)
    m = max_length(link, 85.0, hi_m=30.0)
    assert 5.0 < m["max_length_m"] < 30.0
    assert m["limited_by"]
    lim = limiting_element(link)
    assert lim[0]["gain_db"] >= lim[-1]["gain_db"]
    tc = temperature_ceiling(link)
    assert tc["temperature_ceiling_c"] is None or tc["temperature_ceiling_c"] >= 85.0
    cb = connector_budget(link, Connector(z_diff=88.0, asym_c=0.02), max_n=4)
    assert cb["table"][0]["n_inline"] == 0
    assert cb["max_inline_connectors"] is None or cb["max_inline_connectors"] >= 0


def test_monte_carlo_and_sensitivity(measured_10m):
    link = make_link(measured_10m, 6.0, 85.0)
    mc = monte_carlo(link, Tolerances(), n=12, seed=1, phy="1000base-t1")
    assert 0.0 <= mc.p_pass <= 1.0 and mc.margins.size == 12 and mc.eye_heights.size == 12
    assert np.isfinite(mc.margins).all()
    s = sensitivity(link, Tolerances(), phy="1000base-t1")
    assert {r["parameter"] for r in s} == set(PARAMS)
    a = next(r for r in s if r["parameter"] == "cable_a")
    # a +1 sigma on the sqrt(f) loss lowers the equalised eye, -1 sigma raises it
    assert a["eye_plus_v"] < 0 < a["eye_minus_v"]
