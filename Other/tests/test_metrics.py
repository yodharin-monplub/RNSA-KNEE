import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_models import bootstrap_auc


class MetricTests(unittest.TestCase):
    def test_bootstrap_auc_matches_sklearn_with_ties(self):
        y = np.array([[0,0],[1,0],[0,1],[1,1],[1,0]], dtype=float)
        p = np.array([[.2,.5],[.9,.5],[.2,.8],[.6,.6],[.6,.1]])
        draws = np.array([[0,1,2,3,4],[0,0,1,3,4],[0,1,1,4,4]])
        expected = []
        for draw in draws:
            expected.append(np.mean([roc_auc_score(y[draw,j],p[draw,j]) for j in range(2)
                                      if np.unique(y[draw,j]).size==2]))
        np.testing.assert_allclose(bootstrap_auc(y,p,draws), expected)


if __name__ == '__main__':
    unittest.main()
