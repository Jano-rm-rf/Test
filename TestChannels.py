custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)
from mmpretrain.models import SparseResNet                         
import torch
self = SparseResNet(depth=50, stem_channels=64, norm_cfg=dict(type='SparseSyncBatchNorm2d'), out_indices=(0, 1, 2, 3))
self.eval()
inputs = torch.rand(1, 3, 224, 224)
level_outputs = self.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))
