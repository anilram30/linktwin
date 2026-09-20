import json
from pathlib import Path

import pytest

from linktwin.cli import main
from linktwin.demo import write_examples
from linktwin.harness import load_harness


@pytest.fixture(scope="module")
def examples(tmp_path_factory):
    out = tmp_path_factory.mktemp("ex")
    return out, write_examples(out)


def test_harness_from_touchstone_and_coefficients(examples):
    out, written = examples
    for key in ("camera-link", "coefficients-link"):
        h = load_harness(written[key])
        assert h.link.cable_length_m == 12.0 and h.link.n_connectors == 3 and h.temperature_c == 85.0
        assert h.sources[0]["source"] in ("touchstone", "coefficients")
        r = h.link.evaluate()
        assert r.verdict == "PASS"


def test_cli_commands(examples, capsys):
    out, written = examples
    hz = str(written["camera-link"])
    assert main(["evaluate", hz]) == 0
    assert "verdict PASS" in capsys.readouterr().out
    assert main(["evaluate", hz, "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["evaluation"]["verdict"] == "PASS"
    assert main(["eye", hz, "--no-sim", "--fig", str(out / "eye.png")]) == 0
    assert (out / "eye.png").exists()
    assert main(["limiting", hz]) == 0
    assert main(["montecarlo", hz, "-n", "4", "--fig", str(out / "mc.png")]) == 0
    assert (out / "mc.png").exists()
    assert main(["maxlength", hz, "--temperatures", "85"]) == 0
    assert "max length" in capsys.readouterr().out


def test_production_source_if_available(examples, tmp_path):
    pytest.importorskip("cableanalytics")
    ds = Path(__file__).resolve().parents[1] / "examples" / "production_dataset.csv"
    if not ds.exists():
        pytest.skip("no production data set shipped")
    from linktwin.harness import cable_from_production
    seg, info = cable_from_production(ds, "S0005", 15.0, 85.0)
    assert info["source"] == "production" and info["u_rel"]["a"] > 0
    assert seg.model.c0_d > 0 and 0.6 < seg.model.nvp_d < 0.85
