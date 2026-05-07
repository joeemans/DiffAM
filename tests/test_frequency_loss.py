import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from losses.frequency_loss import FourierCharbonnierLoss


@unittest.skipUnless(torch is not None, "torch is not installed")
class FourierCharbonnierLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.loss_fn = FourierCharbonnierLoss(cutoff=0.15, eps=1e-3)
        self.masked_loss_fn = FourierCharbonnierLoss(
            cutoff=0.15,
            eps=1e-3,
            mask_mode="face_union",
        )

    def test_identical_tensors_are_near_zero(self):
        image = torch.zeros(2, 3, 32, 32)
        loss = self.loss_fn(image, image)
        self.assertLess(loss.item(), 1e-7)

    def test_high_frequency_change_is_penalized_more_than_smooth_shift(self):
        base = torch.zeros(1, 3, 32, 32)
        smooth = base + 0.05

        pattern = (
            (torch.arange(32).unsqueeze(1) + torch.arange(32).unsqueeze(0)) % 2
        ).float()
        pattern = pattern * 2.0 - 1.0
        checker = base + 0.05 * pattern.unsqueeze(0).unsqueeze(0)

        smooth_loss = self.loss_fn(smooth, base)
        checker_loss = self.loss_fn(checker, base)
        self.assertGreater(checker_loss.item(), smooth_loss.item())

    def test_backward_produces_finite_gradients(self):
        pred = torch.tanh(torch.randn(1, 3, 32, 32)).requires_grad_(True)
        ref = torch.tanh(torch.randn(1, 3, 32, 32))

        loss = self.loss_fn(pred, ref)
        loss.backward()

        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all().item())

    def test_identical_tensors_are_near_zero_with_face_union_mask(self):
        image = torch.zeros(2, 3, 32, 32)
        mask = torch.ones(2, 32, 32)
        loss = self.masked_loss_fn(image, image, mask=mask)
        self.assertLess(loss.item(), 1e-7)

    def test_face_union_zero_mask_returns_zero(self):
        pred = torch.tanh(torch.randn(1, 3, 32, 32))
        ref = torch.tanh(torch.randn(1, 3, 32, 32))
        zero_mask = torch.zeros(1, 32, 32)

        loss = self.masked_loss_fn(pred, ref, mask=zero_mask)
        self.assertLess(loss.item(), 1e-7)

    def test_masked_mode_focuses_on_masked_region(self):
        base = torch.zeros(1, 3, 32, 32)
        pred = base.clone()
        mask = torch.zeros(1, 32, 32)
        mask[:, 8:24, 8:24] = 1.0

        pattern = (
            (torch.arange(16).unsqueeze(1) + torch.arange(16).unsqueeze(0)) % 2
        ).float()
        pattern = pattern * 2.0 - 1.0
        pred[:, :, 8:24, 8:24] = 0.05 * pattern.unsqueeze(0).unsqueeze(0)

        masked_loss = self.masked_loss_fn(pred, base, mask=mask)
        unmasked_loss = self.masked_loss_fn(pred, base, mask=torch.zeros_like(mask))

        self.assertGreater(masked_loss.item(), 0.0)
        self.assertLess(unmasked_loss.item(), 1e-7)


if __name__ == "__main__":
    unittest.main()
