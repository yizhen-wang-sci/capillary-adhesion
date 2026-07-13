"""
Tests of the self-affine roughness model.
"""

import numpy as np
import pytest

from a_package.model.roughness import SelfAffineRoughness, psd_to_height


@pytest.fixture
def roughness():
    return SelfAffineRoughness(C0=1.0, H=0.8, qR=2 * np.pi * 4, qS=2 * np.pi * 32)


def test_psd_at_zero_is_zero(roughness):
    """The PSD evaluated at wavenumber 0 must be 0 (below qT)."""
    q = np.linspace(0., 9., 10)
    psd = roughness.mapto_isotropic_psd(q)
    assert psd[np.isclose(q, 0)] == 0.0


def test_psd_to_height_has_zero_mean(roughness):
    """The height field obtained via psd_to_height should have zero mean."""
    # Build a 2D wavevector grid
    n = 16
    L = 1.0
    q = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    qx, qy = np.meshgrid(q, q, indexing="ij")
    wavevector = np.stack([qx, qy], axis=-1)

    psd = roughness.mapto_isotropic_psd(wavevector, component_axis=-1)
    height = psd_to_height(psd, seed=None)

    assert np.isclose(np.mean(height), 0.0, atol=1e-12)
