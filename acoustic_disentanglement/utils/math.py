import torch
import torch.nn as nn
import numpy as np

# Pytorch kld loss, from wnhsu's pytorch scalable fhvae implementation
def kld_fn(p_mu, p_logvar, q_mu, q_logvar, normalize_dim=True):
    p_var = p_logvar.exp()
    q_var = q_logvar.exp()
    sq_mu_diff = (p_mu - q_mu).pow(2)
    kld = -0.5 * (1 + p_logvar - q_logvar - (sq_mu_diff + p_var) / q_var)
    kld = torch.sum(kld.view(kld.size(0), -1), 1)
    if normalize_dim:
        kld = torch.mean(kld.view(kld.size(0), -1), 1)
    else:
        kld = torch.sum(kld.view(kld.size(0), -1), 1)
    return kld
    
log2pi = np.log(2. * np.pi)

# Pytorch log_gauss function, from wnhsu's pytorch scalable fhvae implementation
def log_gauss_fn(x, mu, logvar, normalize_dim=True):
    var = logvar.exp()
    sq_mu_diff = (x - mu).pow(2)
    log_p = -0.5 * (log2pi + logvar + sq_mu_diff / var)
    if normalize_dim:
        log_p = torch.mean(log_p.view(log_p.size(0), -1), 1)
    else:
        log_p = torch.sum(log_p.view(log_p.size(0), -1), 1)
    return log_p

# Inverse variance weighting for taking average of gaussian distributions
def gaussian_normalization(means, variances):
    variance = np.reciprocal(np.sum(np.reciprocal(np.exp(variances)), 0))
    mean = np.sum(means * np.reciprocal(np.exp(variances)), 0) * variance
    return mean, variance