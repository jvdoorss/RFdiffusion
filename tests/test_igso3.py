from math import sqrt

import torch
from scipy.spatial.transform import Rotation
from torch import Tensor
from quatorch import Quaternion


### Original, scipy-Rotations based implementation

def hat(v: Tensor) -> Tensor:
    '''hat map from vector space R^3 to Lie algebra so(3)'''
    hat_v = torch.zeros([v.shape[0], 3, 3])
    hat_v[:, 0, 1], hat_v[:, 0, 2], hat_v[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    return hat_v + -hat_v.transpose(2, 1)

def Log(R: Tensor) -> Tensor: 
    '''Logarithmic map from SO(3) to R^3 (i.e. rotation vector)'''
    return torch.tensor(Rotation.from_matrix(R.numpy()).as_rotvec())  

def log(R: Tensor) -> Tensor: 
    '''logarithmic map from SO(3) to so(3), this is the matrix logarithm'''
    return hat(Log(R))

def Exp(A: Tensor) -> Tensor: 
    '''
    Exponential map from vector space of so(3) to SO(3), this is the matrix
    exponential combined with the "hat" map
    '''
    return torch.tensor(Rotation.from_rotvec(A.numpy()).as_matrix())

def Omega(R: Tensor) -> Tensor: 
    '''Angle of rotation SO(3) to R^+'''
    return torch.norm(log(R).reshape(*R.shape[:-2],-1), dim=-1)/sqrt(2.)

def f_igso3(omega: Tensor, t: float, L : int = 2000) -> Tensor:
    """Truncated sum of IGSO(3) distribution."""
    ls = torch.arange(L)[None]  # of shape [1, L]
    coeff = (2*ls + 1) * torch.exp(-ls*(ls+1)*t/2)
    arg1 = omega[:, None]*(ls+1/2)
    arg2 = omega[:, None]/2
    return ( coeff * torch.sin(arg1) / torch.sin(arg2)).sum(dim=-1)

def d_logf_d_omega(omega: Tensor, t: float, L: int = 2000) -> Tensor:
    '''torch backpropagated derivative of f_igso3 above'''
    omega_diff = torch.tensor(omega, requires_grad=True)
    log_f = torch.log(f_igso3(omega_diff, t, L))
    return torch.autograd.grad(log_f.sum(), omega_diff)[0]

def d_f_d_omega(omega: Tensor, t: float, L: int = 2000) -> Tensor:
    '''Explicit derivative of f_igso3 above'''
    ls = torch.arange(L)[None]  # of shape [1, L]
    coeff = (2*ls + 1) * torch.exp(-ls*(ls+1)*t/2)
    arg1 = omega[:, None]*(ls+1/2)
    arg2 = omega[:, None]/2
    return 0.5 * (coeff * ((2*ls + 1) * torch.cos(arg1)/torch.sin(arg2) - torch.sin(arg1) * torch.cos(arg2)/ torch.sin(arg2)**2)).sum(-1)

def d_logf_d_omega_explicit(omega: Tensor, t: float, L: int = 2000) -> Tensor:
    '''Explicit derivative of f_igso3 above'''
    return d_f_d_omega(omega,t,L) / f_igso3(omega,t,L)

def test_geometric():
    '''Verify equivalence of matrix and quaternion based conversions'''
    Q = Quaternion(torch.randn(10,4)).normalize()
    R = Q.to_rotation_matrix()

    log_r = Log(R).to(R.dtype)
    log_q = 2 * Q.log()
    assert torch.allclose(log_q[...,0],torch.zeros_like(log_q[...,0]),atol=1.e-6)
    modulate = log_q.norm(dim=-1,keepdim=True) > torch.pi
    log_q = torch.where(modulate, log_q - 2 * torch.pi * log_q.normalize(), log_q)
    assert torch.allclose(log_r, log_q[...,1:])

    omega_r = Omega(R)
    axis, omega_q = Q.to_axis_angle()
    flip = omega_q > torch.pi
    # account for ambiguity
    axis = torch.where(flip.unsqueeze(-1),-axis,axis)
    omega_q = torch.where(flip, 2 * torch.pi - omega_q, omega_q)
    assert torch.allclose(omega_r,omega_q)

    exp_r = Exp(axis * omega_q.unsqueeze(-1)).to(R.dtype)
    exp_q = Quaternion.from_axis_angle(axis,omega_q)
    assert torch.allclose(exp_r, R,atol=1.e-6)
    assert torch.all(torch.isclose(exp_q, Q, atol=1.e-6) | torch.isclose(exp_q, -Q, atol=1.e-6))

def test_gradient():
    omega = torch.linspace(0, torch.pi, 11)[1:]
    ts = 10 ** torch.linspace(0.1,1.1,10)
    grad_implicit = torch.stack([d_logf_d_omega(omega,t) for t in ts])
    grad_explicit = torch.stack([d_logf_d_omega_explicit(omega,t) for t in ts])
    assert torch.allclose(grad_implicit,grad_explicit, atol=1.e-3)

if __name__ == '__main__':
    test_geometric()
    test_gradient()