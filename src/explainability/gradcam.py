"""Post-hoc Grad-CAM for the black-box diagnostic baseline.

IMPORTANT: Grad-CAM is a post-hoc *saliency* baseline. It is NOT an
intrinsic explanation of the model's decision and does not reflect the
clinically grounded, spatially faithful mechanism developed in later phases.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from matplotlib import cm
from matplotlib import pyplot as plt

from src.data.transforms import unnormalize
from src.models.encoder import ImageEncoder


def find_last_conv(module: nn.Module) -> Optional[nn.Module]:
    """Return the last Conv2d module found by reverse traversal."""
    found = []

    def visit(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            found.append(m)
        for child in m.children():
            visit(child)

    visit(module)
    return found[-1] if found else None


class GradCAM:
    """Grad-CAM over a target convolutional layer.

    Uses forward/backward hooks to obtain activations and gradients, then
    weights the activation channels by the average gradient (Class
    Activation Mapping).
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        if target_layer is None:
            target_layer = find_last_conv(model.encoder.features)
        if target_layer is None:
            raise ValueError("No Conv2d target layer found for Grad-CAM.")
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out) -> None:
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(
        self, input_tensor: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Generate a [0, 1] CAM mask for a single image (batch of 1)."""
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        # Frozen backbones have no learnable parameters, so gradients would
        # not flow into the feature activations. Clone + require grad to
        # guarantee the backward hooks fire (weights stay untouched).
        input_tensor = input_tensor.detach()
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        self.model.zero_grad()
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[1]

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=False)

        activations = self.activations[0]  # [C, H, W]
        gradients = self.gradients[0]      # [C, H, W]

        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        cam = (weights * activations).sum(dim=0)            # [H, W]
        cam = F.relu(cam)
        cam = cam.unsqueeze(0).unsqueeze(0)                 # [1, 1, H, W]
        cam = F.interpolate(
            cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear", align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        lo, hi = float(cam.min()), float(cam.max())
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        else:
            cam = np.zeros_like(cam)
        return cam

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def _to_display_image(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """Normalized [1,3,H,W] tensor -> uint8 RGB array in [0, 255]."""
    img = unnormalize(tensor.detach().cpu(), mean=mean, std=std)
    img = img[0].clamp(0, 1).numpy()          # [3, H, W]
    img = np.transpose(img, (1, 2, 0))        # [H, W, 3]
    return (img * 255.0).astype(np.uint8)


def visualize_cam(
    input_tensor: torch.Tensor,
    cam: np.ndarray,
    save_path: str | Path,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    title: str = "Grad-CAM",
    dpi: int = 150,
) -> None:
    """Save a three-panel figure: original, Grad-CAM heatmap, overlay."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    original = _to_display_image(input_tensor, mean, std)
    heatmap = (cm.jet(cam)[..., :3] * 255.0).astype(np.uint8)
    overlay = (0.5 * original.astype(np.float32) + 0.5 * heatmap.astype(np.float32)).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_gradcam_examples(
    model: nn.Module,
    dataset,
    device: torch.device,
    out_dir: str | Path,
    num_examples: int = 8,
    target_layer: Optional[nn.Module] = None,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> list:
    """Generate Grad-CAM figures for the first ``num_examples`` dataset items.

    Returns the list of written file paths.
    """
    out_dir = Path(out_dir)
    written = []
    model.eval()
    cam = GradCAM(model, target_layer=target_layer)
    for idx in range(min(num_examples, len(dataset))):
        item = dataset[idx]
        x = item["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[1]
            class_idx = int(logits.argmax(dim=1).item())
        # Grad-CAM needs gradients, so run outside torch.no_grad().
        mask = cam.generate(x, class_idx=class_idx)
        image_id = item["image_id"]
        path = out_dir / f"{image_id}_class{class_idx}_gradcam.png"
        visualize_cam(
            x, mask, path, mean=mean, std=std,
            title=f"Grad-CAM | {image_id} | argmax class {class_idx}",
        )
        written.append(path)
    cam.close()
    return written
