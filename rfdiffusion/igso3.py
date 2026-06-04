"""
SO(3) diffusion methods.

Tools to compute an Isotropic Gaussian distribution on SO(3).
"""
from math import log10

import torch
from torch import Tensor
from quatorch import Quaternion


def quaternion_from_rot_vector(axis_angle: Tensor) -> Quaternion:
    angle = axis_angle.norm()
    axis = axis_angle / angle.unsqueeze(-1)
    return Quaternion.from_axis_angle(axis,angle)

L_default = 2000

def f_igso3(omega: Tensor, t: Tensor | float, L : int = L_default) -> Tensor:
    """Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here, sigma =
    sqrt(2) * eps, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=sigma^2 when defined for the canonical inner product on SO3,
    <u, v>_SO3 = Trace(u v^T)/2

    Args:
        omega: i.e. the angle of rotation associated with rotation matrix
        t: variance parameter of IGSO(3), maps onto time in Brownian motion
        L: Truncation level

    Returns:
        2D tensor of shape (len(t), len(omega))  ((1, len(omega)) if t is float)
    """

    ls = torch.arange(L).reshape(1,1,-1)  # of shape [1, 1, L]
    omega = omega.reshape(1,-1,1) # shape [1,*,1]
    t = t.reshape(-1,1,1) if isinstance(t, Tensor) else omega.new_full((1,1,1),t) # shape [*,1,1]
    return ((2*ls + 1) * torch.exp(-ls*(ls+1)*t/2) *
             torch.sin(omega*(ls+1/2)) / torch.sin(omega/2)).sum(dim=-1)

def d_f_d_omega(omega: Tensor, t: Tensor | float, L: int = 2000) -> Tensor:
    '''Explicit derivative of f_igso3 above'''
    ls = torch.arange(L).reshape(1,1,-1)  # of shape [1, 1, L]
    omega = omega.reshape(1,-1,1) # shape [1,*,1]
    t = t.reshape(-1,1,1) if isinstance(t, Tensor) else omega.new_full((1,1,1),t) # shape [*,1,1]
    coeff = (2*ls + 1) * torch.exp(-ls*(ls+1)*t/2)
    arg1 = omega*(ls+1/2)
    arg2 = omega/2
    return 0.5 * (coeff * ((2*ls + 1) * torch.cos(arg1)/torch.sin(arg2) - torch.sin(arg1) * torch.cos(arg2)/ torch.sin(arg2)**2)).sum(-1)

def d_logf_d_omega(omega: Tensor, t: Tensor | float, L: int = 2000) -> Tensor:
    '''Explicit derivative of f_igso3 above'''
    return d_f_d_omega(omega,t,L) / f_igso3(omega,t,L)

def d_logf_d_omega_old(omega: Tensor, t: Tensor | float, L: int = L_default) -> Tensor:
    omega_diff = torch.tensor(omega, requires_grad=True)
    log_f = torch.log(f_igso3(omega_diff, t, L))
    return torch.autograd.grad(log_f, omega_diff)[0]

def igso3_density(Qt: Quaternion, t: Tensor | float, L: int = L_default):
    '''IGSO3 density with respect to the volume form on SO(3)'''
    _, omega = Qt.to_axis_angle()
    return f_igso3(omega, t, L).numpy()

def igso3_density_angle(omega: Tensor, t: float | Tensor, L: int = L_default) -> Tensor: 
    return f_igso3(omega, t, L) * (1-torch.cos(omega)) / torch.pi

def calculate_igso3(*, num_sigma: int, num_omega: int, min_sigma: float, max_sigma: float) -> dict[str,Tensor]:
    """calculate_igso3 pre-computes numerical approximations to the IGSO3 cdfs
    and score norms and expected squared score norms.

    Args:
        num_sigma: number of different sigmas for which to compute igso3
            quantities.
        num_omega: number of point in the discretization in the angle of
            rotation.
        min_sigma, max_sigma: the upper and lower ranges for the angle of
            rotation on which to consider the IGSO3 distribution.  This cannot
            be too low or it will create numerical instability.
    """
    # Discretize omegas for calculating CDFs. Skip omega=0.
    discrete_omega = torch.linspace(0, torch.pi, num_omega+1)[1:]

    # Exponential noise schedule.  This choice is closely tied to the
    # scalings used when simulating the reverse time SDE. For each step n,
    # discrete_sigma[n] = min_eps^(1-n/num_eps) * max_eps^(n/num_eps)
    discrete_sigma = 10 ** torch.linspace(log10(min_sigma), log10(max_sigma), num_sigma + 1)[1:]

    # Compute the pdf and cdf values for the marginal distribution of the angle
    # of rotation (which is needed for sampling)
    pdf_vals = igso3_density_angle(discrete_omega, discrete_sigma**2)
    cdf_vals = pdf_vals.cumsum(dim=-1) / num_omega * torch.pi

    # Compute the norms of the scores.  This are used to scale the rotation axis when
    # computing the score as a vector.
    score_norm = d_logf_d_omega(discrete_omega, discrete_sigma**2)

    # Compute the standard deviation of the score norm for each sigma
    exp_score_norms = torch.sqrt(torch.sum(score_norm**2 * pdf_vals, dim=1) / torch.sum(pdf_vals, dim=1))
    return {
        'cdf': cdf_vals,
        'score_norm': score_norm,
        'exp_score_norms': exp_score_norms,
        'discrete_omega': discrete_omega,
        'discrete_sigma': discrete_sigma,
    }
