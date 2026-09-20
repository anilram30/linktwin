import numpy as np
from cablecheck.mixedmode import PortMap, to_mixed_mode

from linktwin.elements import CableSegment, Connector, PcbTrace
from linktwin.link import Link
from linktwin.validate import truth_pair, validate_cascade, validate_scaling

PMAP = PortMap.single_pair()


def il_rl(net):
    mm = to_mixed_mode(net, PMAP)
    n, fa = ("d", "A", "near"), ("d", "A", "far")
    return -20 * np.log10(np.abs(mm.param(fa, n))), -20 * np.log10(np.abs(mm.param(n, n)) + 1e-30)


def test_segment_reproduces_its_own_measurement(measured_10m, f):
    seg = CableSegment.from_network(measured_10m, 10.0, PMAP, 10.0, 23.0, 23.0)
    il_m, rl_m = il_rl(measured_10m)
    il_t, rl_t = il_rl(seg.network(f))
    m = f > 5e6
    assert np.max(np.abs(il_m - il_t)[m]) < 0.05
    # a real piece reflects differently from its two ends; the uniform-line twin carries the symmetric part
    assert abs(rl_m.min() - rl_t.min()) < 3.5
    assert abs(f[np.argmin(rl_m)] - f[np.argmin(rl_t)]) < 5e6


def test_segment_scales_with_length_and_temperature(measured_10m, f, base_spec):
    for L, T in ((3.0, 23.0), (25.0, 23.0), (15.0, 125.0)):
        seg = CableSegment.from_network(measured_10m, 10.0, PMAP, L, T, 23.0)
        il_t, _ = il_rl(truth_pair(L, T, f, base_spec))
        il_p, _ = il_rl(seg.network(f))
        assert np.max(np.abs(il_t - il_p)[f > 5e6]) < 0.2, (L, T)


def test_stack_temperature_convention(measured_10m, f, base_spec):
    """With skin_temperature=False the twin matches a truth whose skin resistance does not move with temperature."""
    seg = CableSegment.from_network(measured_10m, 10.0, PMAP, 25.0, 125.0, 23.0, skin_temperature=False)
    il_t, _ = il_rl(truth_pair(25.0, 125.0, f, base_spec, skin_scaling=False))
    il_p, _ = il_rl(seg.network(f))
    assert np.max(np.abs(il_t - il_p)[f > 5e6]) < 0.15
    # and the physical law predicts more loss at temperature
    seg2 = CableSegment.from_network(measured_10m, 10.0, PMAP, 25.0, 125.0, 23.0)
    il_p2, _ = il_rl(seg2.network(f))
    assert il_p2[-1] > il_p[-1] + 1.0


def test_extrapolation_is_smooth_and_causal(measured_10m):
    seg = CableSegment.from_network(measured_10m, 10.0, PMAP, 10.0, 23.0, 23.0)
    f2 = np.linspace(1e6, 2.5e9, 2500)
    il, rl = il_rl(seg.network(f2))
    assert np.all(np.diff(il[f2 > 5e6]) > -0.02)          # monotone within the band-edge taper tolerance
    assert il[-1] > il[np.argmin(np.abs(f2 - 600e6))] * 1.8   # keeps growing beyond the band


def test_connector_is_passive_reciprocal_and_converts_modes():
    f = np.linspace(1e6, 1e9, 200)
    c = Connector(z_diff=88.0, asym_c=0.02)
    net = c.network(f)
    assert np.max(net.passivity_violation()) < 1e-6
    assert np.max(net.reciprocity_error()) < 1e-9
    mm = to_mixed_mode(net, PMAP)
    lcl = -20 * np.log10(np.abs(mm.param(("c", "A", "near"), ("d", "A", "near"))))
    assert 30 < lcl.min() < 60
    c0 = Connector(asym_c=0.0)
    mm0 = to_mixed_mode(c0.network(f), PMAP)
    assert np.max(np.abs(mm0.param(("c", "A", "near"), ("d", "A", "near")))) < 1e-9


def test_link_cascade_matches_sum_of_losses(measured_10m, f):
    a = CableSegment.from_network(measured_10m, 10.0, PMAP, 4.0, 23.0, 23.0, name="a")
    b = CableSegment.from_network(measured_10m, 10.0, PMAP, 8.0, 23.0, 23.0, name="b")
    whole = CableSegment.from_network(measured_10m, 10.0, PMAP, 12.0, 23.0, 23.0, name="w")
    il_ab, _ = il_rl(Link([a, b]).network(f))
    il_w, _ = il_rl(whole.network(f))
    assert np.max(np.abs(il_ab - il_w)[f > 5e6]) < 0.05


def test_link_evaluate_gives_verdict_and_rows(measured_10m):
    link = Link([PcbTrace(), Connector(name="h1"), CableSegment.from_network(measured_10m, 10.0, PMAP, 12.0, 85.0, 23.0),
                 Connector(name="h2"), PcbTrace()], "1000base-t1-link-segment")
    r = link.evaluate()
    assert r.verdict in ("PASS", "FAIL")
    quantities = {row["quantity"] for row in r.rows if row["kind"]}
    assert {"insertion_loss", "return_loss", "lcl", "lctl"} <= quantities
    assert "impedance_fitted" not in quantities
    assert r.length_m == 12.0 and r.n_connectors == 2 and r.temperature_c == 85.0
    assert r.il_db["600MHz"] > r.il_db["100MHz"] > 0


def test_twin_harness_agrees_with_truth_harness():
    v = validate_cascade()
    assert v["same_verdict"]
    assert abs(v["headline_diff_db"]) < 0.1
    assert abs(v["per_quantity"]["insertion_loss"]["diff_db"]) < 0.1


def test_validate_scaling_summary():
    v = validate_scaling(lengths=(3.0, 25.0), temperatures=(23.0, 125.0))
    assert max(r["il_max_err_db"] for r in v["rows"]) < 0.2
    assert abs(v["extrapolation"]["err_at_2_5ghz_db"]) < 0.3
