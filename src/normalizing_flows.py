import torch.nn as nn
import torch


class PlanarFlow(nn.Module):
    """
    A single planar flow, computes T(x) and log(det(jac_T)))
    """
    def __init__(self, D):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.Tensor(1, D), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(1, D), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.h = torch.tanh
        self.init_params()

    def init_params(self):
        self.w.data.uniform_(-0.01, 0.01)
        self.b.data.uniform_(-0.01, 0.01)
        self.u.data.uniform_(-0.01, 0.01)

    def forward(self, z):
        linear_term = torch.mm(z, self.w.T) + self.b
        return z + self.u * self.h(linear_term)

    def h_prime(self, x):
        """
        Derivative of tanh
        """
        return (1 - self.h(x) ** 2)

    def psi(self, z):
        inner = torch.mm(z, self.w.T) + self.b
        return self.h_prime(inner) * self.w

    def log_det(self, z):
        inner = 1 + torch.mm(self.psi(z), self.u.T)
        return torch.log(torch.abs(inner))


class NormalizingFlow(nn.Module):
    """
    A normalizng flow composed of a sequence of planar flows.
    """
    def __init__(self, D, n_flows=2):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList(
            [PlanarFlow(D) for _ in range(n_flows)])

    def sample(self, base_samples):
        """
        Transform samples from a simple base distribution
        by passing them through a sequence of Planar flows.
        """
        samples = base_samples
        for flow in self.flows:
            samples = flow(samples)
        return samples

    def forward(self, x):
        """
        Computes and returns the sum of log_det_jacobians
        and the transformed samples T(x).
        """
        sum_log_det = 0
        transformed_sample = x

        for i in range(len(self.flows)):
            log_det_i = (self.flows[i].log_det(transformed_sample))
            sum_log_det += log_det_i
            transformed_sample = self.flows[i](transformed_sample)

        return transformed_sample, sum_log_det
