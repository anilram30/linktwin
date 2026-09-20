"""Figures for the report: run the demonstration (or reuse a finished one) and copy its figures here.

    python docs/make_figures.py [--reuse DEMO_DIR] [--archive-job DIR] [--production-dataset CSV]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "figures"
WORK = HERE / "_figwork"

WANTED = {
    "camera-link_link.png": "camera_link.png",
    "camera-link_elements.png": "camera_elements.png",
    "camera-link_eye.png": "camera_eye.png",
    "camera-link_maxlength.png": "camera_maxlength.png",
    "camera-link_mc.png": "camera_mc.png",
    "camera-link_tornado.png": "camera_tornado.png",
    "production-link_link.png": "production_link.png",
    "production-link_mc.png": "production_mc.png",
    "production-link_eye.png": "production_eye.png",
    "archive-link_link.png": "archive_link.png",
    "validation_scaling_physical.png": "validation_scaling_physical.png",
    "validation_scaling_stack.png": "validation_scaling_stack.png",
    "validation_extrapolation.png": "validation_extrapolation.png",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse")
    ap.add_argument("--archive-job")
    ap.add_argument("--production-dataset")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.reuse:
        src = Path(a.reuse)
    else:
        from linktwin.demo import run_demo
        run_demo(WORK, a.archive_job, a.production_dataset, quick=False)
        src = WORK
    for name, dst in WANTED.items():
        p = src / "figures" / name
        if p.exists():
            shutil.copy(p, OUT / dst)
    rec = json.loads((src / "demo_record.json").read_text(encoding="utf-8"))
    (OUT / "record.json").write_text(json.dumps({k: rec[k] for k in ("harnesses", "validation", "seconds")}, indent=1)[:2_000_000], encoding="utf-8")
    print("figures in", OUT)


if __name__ == "__main__":
    main()
