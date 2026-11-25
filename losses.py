import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class TVLoss(nn.Module):
    """Total Variation Regularization to smooth interior regions and eliminate fragmented island noise"""
    def __init__(self, weight=0.02):
        super().__init__()
        self.weight = weight

    def forward(self, pred_prob):
        diff_h = torch.abs(pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :])
        diff_w = torch.abs(pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1])
        tv = diff_h.mean() + diff_w.mean()
        return self.weight * tv

class SobelBoundaryLoss(nn.Module):
    """
    Sobel Boundary Gradient Loss (No extra branches / zero learnable parameters)
    Computes spatial gradient magnitude difference between predicted probability map and target mask.
    Forces sharp probability transitions at anatomical polyp margins, directly suppressing HD95.
    """
    def __init__(self, weight=0.4):
        super().__init__()
        self.weight = weight
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred_prob, target):
        device = pred_prob.device
        if self.sobel_x.device != device:
            self.sobel_x = self.sobel_x.to(device)
            self.sobel_y = self.sobel_y.to(device)

        target_float = target.float()
        # Gradients of prediction
        gx_p = F.conv2d(pred_prob, self.sobel_x, padding=1)
        gy_p = F.conv2d(pred_prob, self.sobel_y, padding=1)
        grad_p = torch.sqrt(gx_p ** 2 + gy_p ** 2 + 1e-7)

        # Gradients of target mask
        gx_t = F.conv2d(target_float, self.sobel_x, padding=1)
        gy_t = F.conv2d(target_float, self.sobel_y, padding=1)
        grad_t = torch.sqrt(gx_t ** 2 + gy_t ** 2 + 1e-7)

        # L1 gradient difference
        loss_grad = F.l1_loss(grad_p, grad_t)
        return self.weight * loss_grad

class HybridLoss(nn.Module):
    def __init__(self, use_tv=True, tv_weight=0.02, use_sobel=True, sobel_weight=0.4):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.use_tv = use_tv
        self.tv_loss = TVLoss(weight=tv_weight)
        self.use_sobel = use_sobel
        self.sobel_loss = SobelBoundaryLoss(weight=sobel_weight)

    def forward(self, pred_seg, target_seg):
        loss_bce = self.bce(pred_seg, target_seg)
        loss_dice = self.dice(pred_seg, target_seg)
        total_loss = loss_bce + loss_dice

        prob = torch.sigmoid(pred_seg)
        if self.use_tv:
            total_loss += self.tv_loss(prob)
        if self.use_sobel:
            total_loss += self.sobel_loss(prob, target_seg)

        return total_loss
