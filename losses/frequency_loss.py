import torch
from torch import nn


class FourierCharbonnierLoss(nn.Module):
    def __init__(self, cutoff=0.15, eps=1e-3, mask_mode="none"):
        super().__init__()
        if not 0.0 <= cutoff < 1.0:
            raise ValueError("cutoff must be in [0, 1).")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        if mask_mode not in {"none", "face_union"}:
            raise ValueError("mask_mode must be 'none' or 'face_union'.")

        self.cutoff = float(cutoff)
        self.eps = float(eps)
        self.mask_mode = mask_mode
        self._weight_cache = {}

    def _get_weight_map(self, spatial_shape, device, dtype):
        height, width = spatial_shape
        key = (height, width, device.type, device.index, str(dtype))
        if key not in self._weight_cache:
            fy = torch.fft.fftfreq(height, d=1.0, device=device)
            fx = torch.fft.rfftfreq(width, d=1.0, device=device)
            grid_y, grid_x = torch.meshgrid(fy, fx, indexing='ij')
            radius = torch.sqrt(grid_y.square() + grid_x.square())
            radius = radius / radius.amax().clamp_min(torch.finfo(radius.dtype).eps)

            weight = torch.clamp(
                (radius - self.cutoff) / (1.0 - self.cutoff),
                min=0.0,
                max=1.0,
            )
            self._weight_cache[key] = weight.unsqueeze(0).unsqueeze(0).to(dtype=dtype)

        return self._weight_cache[key]

    def _prepare_mask(self, mask, pred):
        if mask is None:
            if self.mask_mode == "face_union":
                raise ValueError("mask must be provided when mask_mode is 'face_union'.")
            return None

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() != 4:
            raise ValueError("mask must be 3D or 4D shaped as [B, H, W] or [B, 1, H, W].")

        if mask.shape[1] != 1:
            raise ValueError("mask must have a single channel.")
        if mask.shape[-2:] != pred.shape[-2:]:
            raise ValueError("mask and pred must have the same spatial shape.")
        if mask.shape[0] not in {1, pred.shape[0]}:
            raise ValueError("mask batch size must be 1 or match pred batch size.")

        if mask.shape[0] == 1 and pred.shape[0] != 1:
            mask = mask.expand(pred.shape[0], -1, -1, -1)

        return mask.to(device=pred.device, dtype=pred.dtype).clamp_(0.0, 1.0)

    def forward(self, pred, ref, mask=None):
        if pred.shape != ref.shape:
            raise ValueError("pred and ref must have the same shape.")
        if pred.dim() != 4:
            raise ValueError("pred and ref must be 4D tensors shaped as [B, C, H, W].")

        pred = (pred + 1.0) * 0.5
        ref = (ref + 1.0) * 0.5
        mask = self._prepare_mask(mask, pred)
        if self.mask_mode == "face_union":
            pred = pred * mask
            ref = ref * mask

        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm='ortho')
        ref_fft = torch.fft.rfft2(ref, dim=(-2, -1), norm='ortho')
        diff_mag = torch.abs(pred_fft - ref_fft)

        weight = self._get_weight_map(pred.shape[-2:], pred.device, pred.dtype)
        eps = diff_mag.new_tensor(self.eps)
        charbonnier = torch.sqrt(diff_mag.square() + eps.square()) - eps
        weighted = weight * charbonnier
        denom = weight.sum(dim=(-2, -1)).clamp_min(1e-8)
        return (weighted.sum(dim=(-2, -1)) / denom).mean()
