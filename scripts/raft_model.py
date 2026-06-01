"""Thin wrapper around torchvision's RAFT-large for 128x128 DIC tiles.

The wrapper upsamples inputs to 512x512 so RAFT's 1/8 feature map is 64x64,
runs the network, then downsamples the flow back to 128x128 and rescales by
1/4 to return displacement in the original 128x128 pixel space.

The wrapper expects inputs that are already normalized to ``[-1, 1]`` float32
(matching ``Raft_Large_Weights.DEFAULT.transforms()``). Normalization is the
dataset's responsibility; this wrapper is a pass-through on intensity.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


class RAFTStrainModel(nn.Module):
    """RAFT-large adapted to 128x128 tiles via 4x up/downsample."""

    def __init__(self) -> None:
        super().__init__()
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT)

    def forward(self, a: torch.Tensor, b: torch.Tensor, iters: int = 12) -> dict:
        """Run RAFT on a batch of pre-normalized tile pairs; return {flow, flows}."""
        # 1. Upscale 128x128 -> 512x512 so the internal 1/8 feature map is 64x64.
        a_large = F.interpolate(a, size=(512, 512), mode="bilinear", align_corners=False)
        b_large = F.interpolate(b, size=(512, 512), mode="bilinear", align_corners=False)

        # 2. Forward pass. torchvision's keyword is num_flow_updates, not iters.
        flows_large = self.raft(a_large, b_large, num_flow_updates=iters)
        if not isinstance(flows_large, list):
            flows_large = [flows_large]

        # 3. Downsample and rescale to 128x128 pixel space (divide by 4).
        flows_small = []
        for flow_large in flows_large:
            flow_small = F.interpolate(
                flow_large, size=(128, 128), mode="bilinear", align_corners=False
            )
            flow_small = flow_small / 4.0
            flows_small.append(flow_small)

        final_flow = flows_small[-1]
        return {"flow": final_flow, "flows": flows_small}
