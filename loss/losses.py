import numpy as np
import torch
import torch.nn as nn
# from model.net import tensor_erode, tensor_dilation


class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', avg_factor=None):
        super(BCELoss, self).__init__()
        # self.loss = nn.BCELoss(weight=weight, reduce=reduction, size_average=avg_factor)
        self.loss = nn.BCEWithLogitsLoss(weight=weight, reduce=reduction, size_average=avg_factor)

    def __call__(self, pred, target):
        if target == 1:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)

        return self.loss(pred, target)


class LsLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', avg_factor=None):
        super(LsLoss, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, pred, target):
        if target == 1:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        return self.loss(pred, target)


class LossL1(nn.Module):
    def __init__(self, reduction='mean'):
        super(LossL1, self).__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def __call__(self, input, target):
        return self.loss(input, target)


class LossL2(nn.Module):
    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, input, target):
        return self.loss(input, target)


class TripletLoss(nn.Module):
    def __init__(self, reduce=False):
        super(TripletLoss, self).__init__()
        self.loss = nn.TripletMarginLoss(margin=1.0, p=1, reduce=False, size_average=True)

    def __call__(self, anchor, positive, negative):
        return self.loss(anchor=anchor, positive=positive, negative=negative)


class LossSmoothL1(nn.Module):
    def __init__(self):
        super(LossSmoothL1, self).__init__()
        self.loss = nn.SmoothL1Loss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossCrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super(LossCrossEntropy, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)

    def __call__(self, input, target, weight=None):
        return self.loss(input, target)


class InvertibleLoss(nn.Module):
    def __init__(self, batch_size):
        super(InvertibleLoss, self).__init__()
        self.batch_size = batch_size
        self.loss = nn.MSELoss()
        self.Idt = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).view(1, 3, 3).repeat(self.batch_size, 1, 1)
        if torch.cuda.is_available():
            self.Idt = self.Idt.cuda()

    def __call__(self, Hab, Hba):
        return self.loss(torch.bmm(Hab, Hba), self.Idt)


class Mask_Loss(nn.Module):
    def __init__(self, weight=(1, 1)):
        super(Mask_Loss, self).__init__()
        self.weight = weight
        self.loss = nn.BCELoss()

    def gen_weight(self, h, w):
        interval = self.weight[1] - self.weight[0]
        weight = interval * torch.arange(h) / h + self.weight[0]
        weight = torch.repeat_interleave(weight, w)
        return weight.view(1, 1, h, w)

    def __call__(self, x):
        bs, _, h, w = x.size()
        weight = self.gen_weight(h, w)
        weight = weight.repeat(bs, 1, 1, 1).to(x.device)
        mask_loss = self.loss(x, weight)
        return mask_loss


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.autograd.Variable(torch.Tensor(real_samples.shape[0], 1).to(real_samples.device).fill_(1.0),
                                   requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class TVLoss(nn.Module):
    """
    smooth loss (images)
    """

    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def compute_metrics(data, endpoints, manager):
    metrics = {}
    with torch.no_grad():
        # compute metrics
        B = data["label"].size()[0]
        outputs = np.argmax(endpoints["p"].detach().cpu().numpy(), axis=1)
        accuracy = np.sum(outputs.astype(np.int32) == data["label"].detach().cpu().numpy().astype(np.int32)) / B
        metrics['accuracy'] = accuracy
        return metrics


def ComputeErrH(src, dst, H):
    src_xy1 = torch.cat((src, src.new_ones(1)), -1).view(3, 1)
    src_d = torch.mm(H, src_xy1)
    small = 1e-7
    smallers = 1e-6 * (1.0 - torch.ge(torch.abs(src_d[-1]), small).float())
    src_d = src_d[:2] / (src_d[-1] + smallers)
    tmp_err = torch.norm(src_d - dst.view(-1, 1))
    return tmp_err


def ComputeErrFlow(src, dst, flow):
    src_t = src + flow[int(src[1]), int(src[0])]
    error = torch.linalg.norm(dst - src_t)
    return error


def ComputeErr(src, dst):
    error = torch.linalg.norm(dst - src)
    return error


def compute_eval_results(data_batch, output_batch):
    imgs_full = data_batch["imgs_gray_full"]

    device = imgs_full.device

    pt_set = list(map(eval, data_batch["pt_set"]))
    pt_set = list(map(lambda x: x['matche_pts'], pt_set))

    batch_size, _, img_h, img_w = imgs_full.shape
    flow_f = output_batch["flow_f"]

    errs_m = []
    errs_i = []
    for i in range(batch_size):
        pts = torch.Tensor(pt_set[i]).to(device)
        err = 0
        ide = 0
        for j in range(6):
            p1 = pts[j][0]
            p2 = pts[j][1]
            src, dst = p1, p2
            err += ComputeErrFlow(src=src, dst=dst, flow=flow_f[i])
            ide += ComputeErr(src=src, dst=dst)
        err /= 6
        ide /= 6

        errs_m.append(err)
        errs_i.append(ide)

    eval_results = {"errors_m": errs_m, "errors_i": errs_i}

    return eval_results
