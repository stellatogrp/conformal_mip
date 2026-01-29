import torch


class GaussianKernelLoss:
    def __init__(
            self,
            y_cutoff: float,
            bandwidth:float=1.0,
    ):
        self.y_cutoff = y_cutoff
        self.bandwidth = bandwidth

    def __call__(self, y_pred, y_true):
        diff = torch.abs(self.y_cutoff - y_true)
        diff = (diff / torch.mean(diff))
        kernel = torch.exp(-0.5 * (diff / self.bandwidth) ** 2) / (self.bandwidth * (2 * torch.pi) ** 0.5)
        normalized_kernel = kernel / (torch.mean(kernel) + 1e-8)
        loss = torch.mean(normalized_kernel * (y_pred - y_true) ** 2)
        return loss


class RescaledMSE:
    def __init__(
            self,
            eps: float = 1
    ):
        self.eps = eps

    def __call__(self, y_pred, y_true):
        mse = torch.mean((1 / ((1/1000) * y_true + self.eps)) * (y_pred - y_true) ** 2)
        return mse


class RescaledMSE2:
    def __init__(
            self,
            eps: float = 1,
            reweighting: float = 10
    ):
        self.eps = eps
        self.y_upper = 10*eps
        self.y_lower = (1/10)*eps
        self.reweighting = reweighting

    def __call__(self, y_pred, y_true):
        # reweight y around epsilon
        mask = (y_true <= self.y_upper).float() * (y_true >= self.y_lower).float()
        antimask = (y_true >= self.y_upper).float() + (y_true <= self.y_lower).float()
        n_in_mask = sum(mask) + 1
        n_in_antimask = sum(antimask) + 1
        mask = ((self.reweighting * mask / n_in_mask) + (antimask / n_in_antimask)) / (self.reweighting + 1)
        mse = torch.mean((1 /(torch.abs(y_true) + self.eps)) * mask * torch.abs(y_pred - y_true))
        return mse


class LogMSE:
    def __init__(
            self,
            eps: float = 1e-4
    ):
        self.eps = eps

    def __call__(self, y_pred, y_true):
        log_y_pred = torch.log(y_pred + self.eps)
        log_y_true = torch.log(y_true + self.eps)
        mse = torch.mean((log_y_pred - log_y_true) ** 2)
        return mse


class ClippedMSE:
    def __init__(
            self,
            y_upper: float,
            y_lower: float,
            multiplier: float = 1.0
    ):
        self.y_upper = y_upper
        self.y_lower = y_lower
        self.multiplier = multiplier
    
    def __call__(self, y_pred, y_true):
        assert (y_true >= 0).all()
        log_mse = torch.mean((torch.log(y_pred + 1e-4) - torch.log(y_true + 1e-4)) ** 2)
        mask = (y_true <= self.y_upper).float() * (y_true >= self.y_lower).float()
        mse = torch.mean(mask * (y_pred - y_true) ** 2)
        return self.multiplier * mse + log_mse
