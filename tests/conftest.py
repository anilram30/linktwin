import numpy as np
import pytest
from cablecheck.synth import PairSpec, synthesize_measurement

from linktwin.validate import truth_pair


@pytest.fixture(scope="session")
def f():
    return np.linspace(1e6, 600e6, 600)


@pytest.fixture(scope="session")
def base_spec():
    return PairSpec(ripple_amp=0.004, roughness=0.003, seed=3)


@pytest.fixture(scope="session")
def measured_10m(f, base_spec):
    """A noisy 'measurement' of a 10 m pair at 23 C (the synthesiser of project A plus VNA noise)."""
    return synthesize_measurement(truth_pair(10.0, 23.0, f, base_spec), None, noise_db=-85.0, seed=1, name="pairA 10 m")
