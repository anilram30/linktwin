"""
Harness descriptions (TOML) and the cable sources they can draw on.

    [harness]
    name = "ECU-A to camera"
    standard = "1000base-t1-link-segment"      # a cablecheck limit set
    phy = "1000base-t1"                        # a PHY preset of the eye engine
    temperature_c = 23

    [[element]]                                # in order from the transmitter to the receiver
    type = "pcb"        length_m = 0.05  z_diff = 95
    [[element]]
    type = "connector"  name = "header"  z_diff = 90  length_m = 0.02  asym_c = 0.01
    [[element]]
    type = "cable"      name = "cable-1" length_m = 6.0
      source = "touchstone"  path = "pairA.s4p"  measured_length_m = 10  port_map = "A+near,A-near,A+far,A-far"
    # other cable sources:
    #   source = "archive"       job_dir = "<labauto job directory>"        (length, temperature, trust from the sidecar)
    #   source = "coefficients"  a = 1.7e-5  b = 1.6e-10  nvp = 0.68  z_diff = 100   (dB/m/sqrt(Hz), dB/m/Hz)
    #   source = "production"    dataset = "production_dataset.csv"  sample = "S-0123"  (project E predicts a, b, Z)
    [[element]] ...
    [tolerances]                               # optional, for the Monte Carlo (see uncertainty.Tolerances)

The production source is the point of the whole stack: a cable that exists only as extrusion-line
data (project E) becomes a link element with its prediction interval as its uncertainty.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
from cablecheck.io import read_touchstone

from .elements import CableSegment, Connector, Element, PcbTrace
from .link import Link
from .uncertainty import Tolerances

__all__ = ["load_harness", "Harness", "cable_from_touchstone", "cable_from_archive", "cable_from_production"]


class Harness:
    def __init__(self, name: str, link: Link, phy: str, temperature_c: float, tolerances: Tolerances, sources: list[dict], path: str = ""):
        self.name, self.link, self.phy, self.temperature_c, self.tolerances, self.sources, self.path = name, link, phy, temperature_c, tolerances, sources, path

    def describe(self) -> dict:
        return {"name": self.name, "standard": self.link.cable_type.id, "phy": self.phy, "temperature_c": self.temperature_c,
                "elements": self.link.describe(), "sources": self.sources, "tolerances": self.tolerances.to_dict(), "path": self.path}


def cable_from_touchstone(path: str | Path, measured_length_m: float, new_length_m: float, port_map: str = "A+near,A-near,A+far,A-far",
                          temperature_c: float = 23.0, measured_at_c: float = 23.0, name: str = "cable") -> tuple[CableSegment, dict]:
    net = read_touchstone(path)
    seg = CableSegment.from_network(net, measured_length_m, port_map, new_length_m, temperature_c, measured_at_c, name=name)
    return seg, {"source": "touchstone", "path": str(path), "measured_length_m": measured_length_m, "measured_at_c": measured_at_c}


def cable_from_archive(job_dir: str | Path, new_length_m: float | None = None, temperature_c: float = 23.0, name: str = "cable") -> tuple[CableSegment, dict]:
    """A labauto job directory: the raw pair file, the length and temperature from the sidecar, the trust and hashes for the record."""
    job_dir = Path(job_dir)
    metas = sorted(job_dir.glob("*.s4p.meta.json"))
    if not metas:
        raise FileNotFoundError(f"no *.s4p.meta.json in {job_dir}")
    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    m = meta["measurement"]
    L0 = float(meta["sample"]["length_m"])
    T0 = float(meta["environment"].get("sample_temperature_c", 23.0))
    net = read_touchstone(job_dir / m["file"])
    fx = meta.get("fixture") or {}
    if fx.get("method") == "files" and (fx.get("left") or fx.get("right")):
        # the archive holds raw instrument data; remove the fixture the way project A's pipeline does
        from cablecheck.deembed import deembed
        left = read_touchstone(job_dir / fx["left"]) if fx.get("left") else None
        right = read_touchstone(job_dir / fx["right"]) if fx.get("right") else None
        net = deembed(net, left, right)
    seg = CableSegment.from_network(net, L0, m["port_map"], new_length_m if new_length_m is not None else L0, temperature_c, T0,
                                    name=name or meta["sample"]["sample_id"])
    info = {"source": "archive", "job_dir": str(job_dir), "file": m["file"], "sha256": m.get("sha256"), "sample_id": meta["sample"]["sample_id"],
            "measured_length_m": L0, "measured_at_c": T0, "fixture": fx.get("method", "none"),
            "trust": meta["validation"].get("trust"), "job_trust": meta["validation"].get("job_trust"),
            "calibration": meta.get("calibration", {}).get("id"), "verification": (meta.get("calibration", {}).get("verification") or {}).get("status"),
            "procedure_hash": meta.get("procedure", {}).get("hash"), "site": meta.get("site")}
    man = job_dir / "manifest.json"
    if man.exists():
        # the manifest seals every file of the job with its SHA-256; verify it before trusting the raw file
        try:
            from labauto.archive import Archive
            v = Archive(job_dir.parent).verify(job_dir)
            info["archive_integrity"] = bool(v.get("ok"))
            if not v.get("ok"):
                info["archive_integrity_detail"] = {k: v.get(k) for k in ("modified", "missing")}
        except Exception:  # pragma: no cover
            info["archive_integrity"] = None
    return seg, info


def cable_from_production(dataset: str | Path, sample: str, new_length_m: float, temperature_c: float = 23.0, name: str = "cable") -> tuple[CableSegment, dict]:
    """Project E: fit the manufacturing -> physics correlation on the production data set and predict this sample's
    a, b, Z with 90 % intervals; the segment carries the predicted coefficients and the intervals as its uncertainty."""
    try:
        import pandas as pd
        from cableanalytics.correlate import fit_correlation, predict_coefficients
    except ImportError as e:
        raise ImportError("the production source needs cableanalytics (project E) installed") from e
    df = pd.read_csv(dataset)
    corr = fit_correlation(df)
    row = df[df["sample_id"] == sample]
    if row.empty:
        raise KeyError(f"sample {sample!r} not in {dataset}")
    row = row.iloc[0]
    coef = predict_coefficients(corr, row)
    a, b, z = coef["a"]["value"], coef["b"]["value"], coef["z"]["value"]
    # what the correlation does not predict comes from the design physics of project E: the velocity from the
    # effective permittivity of the (foamed) insulation and the DC floor c0 = 8.686 R_dc / Z_d from the conductor
    try:
        from cableanalytics.physics import ALLOYS, Design, hf_properties
        d = Design(str(row["alloy"]), float(row["d_cond_mm"]), str(row["insulation"]), float(row["foaming"]), float(row["d_ins_mm"]))
        props = hf_properties(d, 20.0)
        nvp = float(props["nvp"])
        r_dc = ALLOYS[d.alloy][0] / (np.pi * (d.d_cond_mm * 1e-3) ** 2 / 4)
        c0 = float(8.686 * r_dc / z)
    except Exception:  # pragma: no cover - older data sets without the design columns
        nvp, c0 = float(row["nvp"]) if "nvp" in row else 0.68, 0.0
    seg = CableSegment.from_coefficients(a, b, nvp, z, new_length_m, temperature_c=temperature_c, name=name,
                                         source=f"production prediction of {sample} (cableanalytics)", c0=c0)
    rel = {k: float((coef[k]["hi90"] - coef[k]["lo90"]) / (2 * 1.645 * coef[k]["value"])) for k in ("a", "b")}
    return seg, {"source": "production", "dataset": str(dataset), "sample": sample, "predicted": {"a": a, "b": b, "z": z, "nvp": nvp, "c0": c0},
                 "design": {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in ("line", "lot", "design", "alloy", "insulation") if k in row},
                 "intervals90": {k: [coef[k]["lo90"], coef[k]["hi90"]] for k in ("a", "b", "z")}, "u_rel": rel,
                 "note": "no measurement of this cable exists; the coefficients come from the manufacturing-to-physics model"}


def load_harness(path: str | Path) -> Harness:
    path = Path(path)
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    h = doc.get("harness", {})
    T = float(h.get("temperature_c", 23.0))
    elements: list[Element] = []
    sources: list[dict] = []
    tol = Tolerances(**doc.get("tolerances", {}))
    for k, e in enumerate(doc.get("element", [])):
        typ = e.get("type")
        name = e.get("name", f"{typ}{k + 1}")
        if typ == "cable":
            src = e.get("source", "coefficients")
            L = float(e["length_m"])
            if src == "touchstone":
                seg, info = cable_from_touchstone(path.parent / e["path"] if not Path(e["path"]).is_absolute() else e["path"], float(e["measured_length_m"]), L,
                                                  e.get("port_map", "A+near,A-near,A+far,A-far"), T, float(e.get("measured_at_c", 23.0)), name)
            elif src == "archive":
                seg, info = cable_from_archive(path.parent / e["job_dir"] if not Path(e["job_dir"]).is_absolute() else e["job_dir"], L, T, name)
            elif src == "production":
                seg, info = cable_from_production(path.parent / e["dataset"] if not Path(e["dataset"]).is_absolute() else e["dataset"], e["sample"], L, T, name)
                if "u_rel" in info:
                    tol.cable_a_rel, tol.cable_b_rel = max(tol.cable_a_rel, info["u_rel"]["a"]), max(tol.cable_b_rel, info["u_rel"]["b"])
            else:
                seg = CableSegment.from_coefficients(float(e["a"]), float(e["b"]), float(e.get("nvp", 0.68)), float(e.get("z_diff", 100.0)), L,
                                                     float(e.get("z_comm", 35.0)), temperature_c=T, name=name)
                info = {"source": "coefficients", **{k: e[k] for k in ("a", "b", "nvp", "z_diff") if k in e}}
            elements.append(seg)
            sources.append({"element": name, **info})
        elif typ == "connector":
            elements.append(Connector(float(e.get("z_diff", 90.0)), float(e.get("length_m", 0.025)), float(e.get("nvp", 0.6)), float(e.get("r_contact_ohm", 0.005)),
                                      float(e.get("l_pin_nh", 1.0)), float(e.get("c_pad_pf", 0.3)), float(e.get("asym_c", 0.01)), float(e.get("loss_db_per_m_at_1ghz", 2.0)),
                                      e.get("z_comm"), name))
        elif typ == "pcb":
            elements.append(PcbTrace(float(e.get("z_diff", 95.0)), float(e.get("length_m", 0.05)), float(e.get("nvp", 0.55)), float(e.get("a", 3e-6)), float(e.get("b", 8e-11)), name))
        else:
            raise ValueError(f"unknown element type {typ!r}")
    link = Link(elements, h.get("standard", "1000base-t1-link-segment"), h.get("name", path.stem), h.get("limits_dirs"))
    return Harness(h.get("name", path.stem), link, h.get("phy", "1000base-t1"), T, tol, sources, str(path))
