import unittest
import numpy as np
import pandas as pd
from src.reference_rank_blend import blend, ID, TARGETS


def frame(ids=("study_c", "study_a", "study_b")):
    d = {ID: list(ids)}
    for target in TARGETS:
        d[target] = [0.15, 0.55, 0.90]
    return pd.DataFrame(d)


class RankBlendTests(unittest.TestCase):
    def test_valid_schema_and_simple_ranks(self):
        a = frame()
        out = blend({"v7": a, "s6": a, "v5": a})
        self.assertEqual(list(out.columns), [ID, *TARGETS])
        np.testing.assert_allclose(out["ACL"], [1 / 3, 2 / 3, 1])

    def test_permuted_rows_align_on_uid(self):
        a = frame()
        out = blend({"v7": a, "s6": a.iloc[::-1], "v5": a.iloc[[1, 2, 0]]})
        np.testing.assert_allclose(out["ACL"], [1 / 3, 2 / 3, 1])

    def test_missing_studies_rejected(self):
        a = frame()
        b = frame(ids=("study_c", "study_a", "other"))
        with self.assertRaisesRegex(ValueError, "study-ID"):
            blend({"v7": a, "s6": b, "v5": a})

    def test_nan_and_schema_rejected(self):
        a = frame()
        bad = a.copy()
        bad.loc[0, "ACL"] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            blend({"v7": a, "s6": bad, "v5": a})
        with self.assertRaisesRegex(ValueError, "columns"):
            blend({"v7": a.drop(columns=["ACL"]), "s6": a, "v5": a})

    def test_duplicates_and_range_rejected(self):
        a = frame()
        dup = a.copy()
        dup.loc[1, ID] = dup.loc[0, ID]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            blend({"v7": a, "s6": dup, "v5": a})
        bad = a.copy()
        bad.loc[0, "ACL"] = 1.1
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            blend({"v7": a, "s6": bad, "v5": a})


if __name__ == "__main__":
    unittest.main()
