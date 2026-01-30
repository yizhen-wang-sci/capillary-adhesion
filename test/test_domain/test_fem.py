import numpy as np

from a_package.domain.fem import LinearFiniteElementPixel


def test_value_interpolation_coefficients():
    fe_pixel = LinearFiniteElementPixel()
    test_pts = np.array([[1/3, 1/3], [2/3, 2/3]])
    coeffs = fe_pixel.compute_value_interpolation_coefficients(test_pts)

    pixel_mapping = np.zeros((test_pts.shape[0], 2, 2))
    for [i_sub_pt, sub_pt_coeffs] in enumerate(coeffs):
        for [i_node, coeff] in sub_pt_coeffs.items():
            pixel_mapping[(i_sub_pt, *i_node)] = coeff

    expected_mapping = np.array([
        [[1/3, 1/3],
         [1/3, 0]],
        [[0, 1/3],
         [1/3, 1/3]]
    ])

    assert np.allclose(pixel_mapping, expected_mapping)
