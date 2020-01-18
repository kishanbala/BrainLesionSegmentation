import torch
from torch import nn
from torch import einsum

class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape  # (batch size,class_num,x,y,z)
        shp_y = gt.shape  # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        # if self.apply_nonlin is not None:
        #     softmax_output = self.apply_nonlin(net_output)
        softmax_output = torch.nn.functional.softmax(net_output,dim=1)

        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = 1 / (einsum("bcxy->bc", y_onehot).type(torch.float32) + 1e-10) ** 2
        intersection: torch.Tensor = w * einsum("bcxy, bcxy->bc", softmax_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxy->bc", softmax_output) + einsum("bcxy->bc", y_onehot))
        divided: torch.Tensor = 1 - 2 * (einsum("bc->b", intersection) + self.smooth) / (
                    einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc

# import torch
# from typing import List
# from torch import Tensor, einsum
#
# class DiceLoss():
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         #self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")
#
#     def __call__(self, probs, target):
#
#         pc = probs.type(torch.float32)
#         tc = target.type(torch.float32)
#
#         intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
#         union: Tensor = (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))
#
#         divided: Tensor = 1 - (2 * intersection + 1e-10) / (union + 1e-10)
#
#         loss = divided.mean()
#
#         return loss