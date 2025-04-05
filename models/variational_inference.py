import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Bayesian Linear Layer using Variational Inference
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters for weight
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))

        # Variational parameters for bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        # Prior standard deviation
        self.prior_std = prior_std

    def forward(self, x):
        # Sample weights and biases using reparameterization trick
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))

        eps_w = torch.randn_like(weight_std)
        eps_b = torch.randn_like(bias_std)

        weight = self.weight_mu + weight_std * eps_w
        bias = self.bias_mu + bias_std * eps_b

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        # Compute KL divergence between variational posterior and prior
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))

        kl_weight = self._kl_div(self.weight_mu, weight_std, self.prior_std)
        kl_bias = self._kl_div(self.bias_mu, bias_std, self.prior_std)

        return kl_weight + kl_bias

    def _kl_div(self, mu_q, std_q, std_p):
        var_q = std_q ** 2
        var_p = std_p ** 2
        kl = torch.log(std_p / std_q) + (var_q + mu_q ** 2) / (2 * var_p) -_***
