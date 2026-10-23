import unittest
import numpy as np
from src.data.geometry import canonical_slice_normal, order_and_sample_slices


class GeometryTests(unittest.TestCase):
    def test_sagittal_canonical_normal_from_reversed_iop(self):
        # cross(Y,Z) = +X; change sign to force negative X
        iop = [0, 1, 0, 0, 0, -1]
        np.testing.assert_allclose(canonical_slice_normal(iop, 'sagittal'), [1, 0, 0])

    def test_sorted_deduplicated_physically_sampled(self):
        iop = [1, 0, 0, 0, 1, 0]  # normal +Z
        positions = [[0, 0, z] for z in list(reversed(range(24)))]
        positions.append([0, 0, 5.0001])  # same physical location to 0.01mm
        idx, zs = order_and_sample_slices(positions, iop, 'axial')
        self.assertEqual(len(idx), 16)
        self.assertTrue(np.all(np.diff(zs) > 0))
        self.assertAlmostEqual(zs[0], 0.)
        self.assertAlmostEqual(zs[-1], 23.)
        self.assertEqual(len(set(idx.tolist())), 16)

    def test_too_few_slices_fail_loudly(self):
        iop = [1, 0, 0, 0, 1, 0]
        with self.assertRaisesRegex(ValueError, 'need 16'):
            order_and_sample_slices([[0, 0, z] for z in range(12)], iop, 'axial')

    def test_invalid_iop_and_oblique_rejected(self):
        with self.assertRaisesRegex(ValueError, 'orthonormal'):
            canonical_slice_normal([1,0,0,1,0,0], 'axial')
        with self.assertRaisesRegex(ValueError, 'too oblique'):
            canonical_slice_normal([0,1,0,0,0,1], 'axial')


if __name__ == '__main__':
    unittest.main()
