import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import numpy as np


# If needed, add the path to the DeepLab repo so `import network` works.
# Example (adjust to where your `modeling.py` and `main.py` live):
# sys.path.append("/home/tzh005/DeepLabV3Plus-Pytorch")
# or whatever the root of that repo is.

from src.external import network  # this is the repo that has modeling.py / main.py
import torch.nn.functional as F


class DeepLabV3PlusMobileNetCityscapes(nn.Module):
    """
    DeepLabV3+ MobileNetV2 trained on Cityscapes (19 classes, OS=16).

    Expects BGR images as tensors and returns logits [B, 19, H, W].
    """

    def __init__(
        self,
        chk_path="exp/0300_DeepLabV3Plus_MobileNet_Cityscapes/best_deeplabv3plus_mobilenet_cityscapes_os16.pth",
        device=None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        chk_path = Path(chk_path)
        print(f"[DeepLabV3PlusMobileNetCityscapes] Loading checkpoint from {chk_path}")

        # 1) Build the model: this calls the factory from modeling.py via network
        self.model = network.deeplabv3plus_mobilenet(
            num_classes=19,
            output_stride=16,
            pretrained_backbone=False,  # we are loading our own weights
        )

        # 2) Load checkpoint (same structure as in main.py)
        state = torch.load(chk_path, map_location=device)

        # In main.py they save as:
        # torch.save({"model_state": model.module.state_dict(), ...})
        if isinstance(state, dict) and "model_state" in state:
            sd = state["model_state"]
        else:
            sd = state  # fallback: raw state_dict

        # No "module." prefix expected, but handle it just in case
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                new_sd[k[len("module."):]] = v
            else:
                new_sd[k] = v

        missing, unexpected = self.model.load_state_dict(new_sd, strict=False)
        print("[DeepLabV3PlusMobileNetCityscapes] missing keys:", missing)
        print("[DeepLabV3PlusMobileNetCityscapes] unexpected keys:", unexpected)

        self.model.to(device)
        self.model.eval()

        # 3) Standard ImageNet mean/std
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        self.register_buffer(
            "mean",
            torch.tensor(mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(std).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, img_b):
        """
        img_b:
          - [B, H, W, 3] or [B, 3, H, W]
          - uint8 or float
          - BGR format (like the rest of your pipeline)

        Returns:
          logits: [B, 19, H, W]
        """
        x = img_b
        if x.dtype != torch.float32:
            x = x.float()

        # [B, H, W, 3] -> [B, 3, H, W]
        if x.dim() == 4 and x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)

        # [0,255] → [0,1] if needed
        if x.max() > 1.0:
            x = x / 255.0

        # BGR → RGB
        x = x[:, [2, 1, 0], :, :]

        device = self.device
        x = x.to(device)
        mean = self.mean.to(device)
        std = self.std.to(device)

        x = (x - mean) / std

        out = self.model(x)
        # If model returns dict (unlikely here, but safe):
        if isinstance(out, dict):
            out = out["out"]

        return out

    @torch.no_grad()
    def forward_multisample(self, image, n_samples: int = 8):
        """
        Bayes-style interface used by ExpSemSegBayes.

        Expected output format:
            {
                'mean': [B, C, H, W]  # class probabilities
                'var':  [B, C, H, W]  # per-class variance (we set it to ~0 here)
            }

        For now we just do a single deterministic forward pass and set var to 0.
        That’s enough to make the evaluation pipeline run.
        """
        self.eval()  # make sure we're in eval mode

        # image is expected as [B, 3, H, W] on the correct device
        logits = self.forward(image)                 # [B, C, H, W]
        prob   = F.softmax(logits, dim=1)            # [B, C, H, W]

        var = torch.zeros_like(prob)                 # no uncertainty for MobileNet

        return {
            "mean": prob,
            "var":  var,
        }
    
    @torch.no_grad()
    def predict_from_pil(self, img_pil: Image.Image) -> torch.Tensor:
        """
        PIL RGB -> logits [1, C, H, W], same as your test.
        """
        img_np = np.array(img_pil)[:, :, ::-1].copy()      # RGB -> BGR
        img_b  = torch.from_numpy(img_np).unsqueeze(0)     # [1, H, W, 3]
        img_b  = img_b.to(self.device)
        logits = self.forward(img_b)
        return logits
