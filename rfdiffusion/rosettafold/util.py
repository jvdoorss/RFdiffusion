import torch

from .chemical import cos_ideal_NCAC

# More complicated version splits error in CA-N and CA-C (giving more accurate CB position)
# It returns the rigid transformation from local frame to global frame
def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
    # N, Ca, C - [B,L, 3]
    # R - [B,L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
    B, L = N.shape[:2]

    v1 = C - Ca
    v2 = N - Ca
    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    u2 = v2 - (torch.einsum("bli, bli -> bl", e1, v2)[..., None] * e1)
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)  # [B,L,3,3] - rotation matrix

    if non_ideal:
        v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)
        cosref = torch.sum(e1 * v2, dim=-1)  # cosine of current N-CA-C bond angle
        costgt = cos_ideal_NCAC.item()
        cos2del = torch.clamp(
            cosref * costgt
            + torch.sqrt((1 - cosref * cosref) * (1 - costgt * costgt) + eps),
            min=-1.0,
            max=1.0,
        )
        cosdel = torch.sqrt(0.5 * (1 + cos2del) + eps)
        sindel = torch.sign(costgt - cosref) * torch.sqrt(1 - 0.5 * (1 + cos2del) + eps)
        Rp = torch.eye(3, device=N.device).repeat(B, L, 1, 1)
        Rp[:, :, 0, 0] = cosdel
        Rp[:, :, 0, 1] = -sindel
        Rp[:, :, 1, 0] = sindel
        Rp[:, :, 1, 1] = cosdel

        R = torch.einsum("blij,bljk->blik", R, Rp)

    return R, Ca
