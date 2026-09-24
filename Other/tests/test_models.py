import unittest

import torch

from knee.models import StudyHead, confidence_bce


class ModelTests(unittest.TestCase):
    def test_absent_slots_do_not_change_predictions(self):
        torch.manual_seed(1)
        x = torch.randn(3, 6, 3, 16)
        mask = torch.ones(3, 6)
        mask[:, 1] = 0
        modified = x.clone(); modified[:, 1] = 999
        for kind in ['linear', 'attention']:
            model = StudyHead(16, kind, 0.2).eval()
            self.assertTrue(torch.allclose(model(x, mask), model(modified, mask), atol=1e-6))

    def test_unknown_targets_have_zero_gradient(self):
        logits = torch.tensor([[0.0, 1.0], [1.0, -1.0]], requires_grad=True)
        targets = torch.tensor([[0.5, 1.0], [0.5, 0.0]])
        weights = 2 * (targets - 0.5).abs()
        loss = confidence_bce(logits, targets, weights)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[:, 0], torch.zeros(2)))
        self.assertGreater(float(logits.grad[:, 1].abs().sum()), 0)

    def test_empty_confidence_is_finite(self):
        logits = torch.randn(2, 12, requires_grad=True)
        loss = confidence_bce(logits, torch.full_like(logits, 0.5), torch.zeros_like(logits))
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


if __name__ == '__main__':
    unittest.main()
