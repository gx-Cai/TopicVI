## Author: CGX
## Time: 2024 06 04
## Description: Sinkhorn algorithm for optimal transport. [tensorized backend]
## Modification of geomloss package for allow cost matrix.

import torch
import numpy as np
from geomloss.utils import scal
from geomloss.sinkhorn_divergence import scaling_parameters, sinkhorn_cost, sinkhorn_loop, log_weights
from geomloss.sinkhorn_samples import softmin_tensorized

__all__ = ["sinkhorn_costmat"]

def sinkhorn_costmat(
    C,
    p=2,
    blur=0.05,
    reach=None,
    diameter=0.1,
    scaling=0.5,
    potentials=False,
):
    debias=False
    device = C.device
    N, M = C.shape
    # assert C.requires_grad

    # Please refer to the comments in this file for more details.
    C_xy = C.unsqueeze(0)  # (B,N,M) torch Tensor
    C_yx = C.unsqueeze(0).transpose(1,2)  # (B,M,N) torch Tensor
    C_xx = None  # (B,N,N) torch Tensor
    C_yy = None  # (B,M,M) torch Tensor

    x = torch.ones((1,N,1), device=device)
    y = torch.ones((1,M,1), device=device)
    a = torch.ones(N, device=device).type_as(x) / N
    b = torch.ones(M, device=device).type_as(y) / M

    # Compute the relevant values of the diameter of the configuration,
    # target temperature epsilon, temperature schedule across itereations
    # and strength of the marginal constraints:
    diameter, eps, eps_list, rho = scaling_parameters(
        x, y, p, blur, reach, diameter, scaling
    )

    # Use an optimal transport solver to retrieve the dual potentials:
    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
        softmin_tensorized,
        log_weights(a),
        log_weights(b),
        C_xx,
        C_yy,
        C_xy,
        C_yx,
        eps_list,
        rho,
        debias=debias,
    )

    # Optimal transport cost:
    return sinkhorn_cost(
        eps,
        rho,
        a,
        b,
        f_aa,
        g_bb,
        g_ab,
        f_ba,
        batch=False,
        debias=debias,
        potentials=potentials,
    )
