"""CPU-only smoke tests use tiny substitutes rather than downloading models."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.cached import CachedStudyDataset, validate_patient_fold_map
from src.models.common import TARGETS, SliceTransformer
from src.models.triplanar_dinov2 import TriPlanarDINOv2
from src.models.triplanar_nfnet import TriPlanarNFNet
from src.training.engine import masked_weighted_bce, train_epoch, macro_roc_auc


class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 8
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
    def forward(self, x):
        return self.conv(x).relu()


class ToyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 12
        self.num_prefix_tokens = 1
        self.conv = nn.Conv2d(3, 12, 4, stride=4)
    def forward_features(self, x):
        y = self.conv(x).flatten(2).transpose(1, 2)
        return torch.cat((y.mean(dim=1, keepdim=True), y), dim=1)


def mini_input():
    torch.manual_seed(17)
    ax = torch.zeros(2, 16, 28, 28)
    cor = torch.zeros_like(ax)
    sag = torch.zeros_like(ax)
    ax[:, :2] = torch.rand(2, 2, 28, 28)
    cor[0, :2] = torch.rand(2, 28, 28)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    return ax, cor, sag, mask


class ResearchModuleTests(unittest.TestCase):
    def test_nfnet_output_and_gradient(self):
        net = TriPlanarNFNet(backbone_factory=ToyCNN, embedding_dim=16, chunk_size=3)
        pred = net(*mini_input())
        self.assertEqual(pred.shape, (2, 12))
        self.assertTrue(torch.isfinite(pred).all())
        pred.sum().backward()
        self.assertIsNotNone(net.encoders['axial'].backbone.conv.weight.grad)
        self.assertIsNone(net.encoders['sagittal'].backbone.conv.weight.grad)

    def test_dinov2_output_and_gradient(self):
        net = TriPlanarDINOv2(backbone_factory=ToyViT, embedding_dim=18, chunk_size=3)
        pred = net(*mini_input())
        self.assertEqual(pred.shape, (2, 12))
        self.assertTrue(torch.isfinite(pred).all())
        pred.square().mean().backward()
        self.assertIsNotNone(net.encoders['axial'].backbone.conv.weight.grad)

    def test_all_missing_is_finite(self):
        net = SliceTransformer(12, heads=3, focal=True)
        h = net(torch.zeros(3, 16, 12), torch.zeros(3, 16, dtype=torch.bool))
        self.assertEqual(h.shape, (3, 12))
        self.assertTrue(torch.isfinite(h).all())
        self.assertTrue(torch.equal(h, torch.zeros_like(h)))

    def test_soft_targets_loss_and_validation(self):
        pred = torch.zeros(2, 12, requires_grad=True)
        y = torch.zeros(2, 12)
        y[0, 0] = .8
        y[1, 0] = float('nan')
        w = torch.tensor([10., 1.])
        loss = masked_weighted_bce(pred, y, w)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        with self.assertRaises(ValueError):
            masked_weighted_bce(pred, y * 0 + 2, w)

    def test_fold_conflict_rejected(self):
        df = pd.DataFrame({'StudyInstanceUID':['a','b','c'],
                           'patient_id':['p','p','q'], 'fold':[0,1,1]})
        with self.assertRaisesRegex(ValueError, 'multiple folds'):
            validate_patient_fold_map(df)

    def test_npz_dataset_and_path_safety(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez_compressed(root/'ax.npz', img=np.full((16,288,288), 128, dtype=np.uint8))
            row = {'StudyInstanceUID':'sample-a','patient_id':'patient-a',
                   'fold':0,'sample_weight':10., 'axial_path':'ax.npz',
                   'coronal_path':None,'sagittal_path':None}
            row.update({k:0. for k in TARGETS})
            data = CachedStudyDataset(pd.DataFrame([row]), root, image_size=28)
            x = data[0]
            self.assertEqual(x[0].shape, (16,28,28))
            self.assertEqual(x[3].tolist(), [1, 0, 0])
            self.assertTrue(torch.isfinite(x[0]).all())
            row['axial_path']='../escaped.npz'
            with self.assertRaisesRegex(ValueError,'escapes'):
                CachedStudyDataset(pd.DataFrame([row]),root,image_size=28)[0]

    def test_macro_auc_valid_only(self):
        y = np.zeros((4,12))
        y[:,0] = [0,0,1,1]
        p = y.copy()
        out, per_class = macro_roc_auc(p, y)
        self.assertAlmostEqual(out, 1.)
        self.assertEqual(list(per_class), ['ACL'])


if __name__ == '__main__':
    unittest.main()
