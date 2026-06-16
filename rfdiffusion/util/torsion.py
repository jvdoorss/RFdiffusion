'''
Utilities module computing (cached) torsions. 
'''

from functools import lru_cache

import torch

from rfdiffusion.rosettafold.chemical import (
    aa2num, torsions, aa2long, ideal_coords, aa2longalt)
from rfdiffusion.rosettafold import rigid_from_3_points

from .geometry import make_frame

def th_ang_v(ab, bc, eps: float = 1e-8):
    def th_norm(x, eps: float = 1e-8):
        return x.square().sum(-1, keepdim=True).add(eps).sqrt()

    def th_N(x, alpha: float = 0):
        return x / th_norm(x).add(alpha)

    ab, bc = th_N(ab), th_N(bc)
    cos_angle = torch.clamp((ab * bc).sum(-1), -1, 1)
    sin_angle = torch.sqrt(1 - cos_angle.square() + eps)
    dih = torch.stack((cos_angle, sin_angle), -1)
    return dih


def th_dih_v(ab, bc, cd):
    def th_cross(a, b):
        a, b = torch.broadcast_tensors(a, b)
        return torch.cross(a, b, dim=-1)

    def th_norm(x, eps: float = 1e-8):
        return x.square().sum(-1, keepdim=True).add(eps).sqrt()

    def th_N(x, alpha: float = 0):
        return x / th_norm(x).add(alpha)

    ab, bc, cd = th_N(ab), th_N(bc), th_N(cd)
    n1 = th_N(th_cross(ab, bc))
    n2 = th_N(th_cross(bc, cd))
    sin_angle = (th_cross(n1, bc) * n2).sum(-1)
    cos_angle = (n1 * n2).sum(-1)
    dih = torch.stack((cos_angle, sin_angle), -1)
    return dih


def th_dih(a, b, c, d):
    return th_dih_v(a - b, b - c, c - d)

def get_tor_mask(seq, torsion_indices, mask_in=None):
    B, L = seq.shape[:2]
    tors_mask = torch.ones((B, L, 10), dtype=torch.bool, device=seq.device)
    tors_mask[..., 3:7] = torsion_indices[seq, :, -1] > 0
    tors_mask[:, 0, 1] = False
    tors_mask[:, -1, 0] = False

    # mask for additional angles
    tors_mask[:, :, 7] = seq != aa2num["GLY"]
    tors_mask[:, :, 8] = seq != aa2num["GLY"]
    tors_mask[:, :, 9] = torch.logical_and(seq != aa2num["GLY"], seq != aa2num["ALA"])
    tors_mask[:, :, 9] = torch.logical_and(tors_mask[:, :, 9], seq != aa2num["UNK"])
    tors_mask[:, :, 9] = torch.logical_and(tors_mask[:, :, 9], seq != aa2num["MAS"])

    if mask_in != None:
        # mask for missing atoms
        # chis
        ti0 = torch.gather(mask_in, 2, torsion_indices[seq, :, 0])
        ti1 = torch.gather(mask_in, 2, torsion_indices[seq, :, 1])
        ti2 = torch.gather(mask_in, 2, torsion_indices[seq, :, 2])
        ti3 = torch.gather(mask_in, 2, torsion_indices[seq, :, 3])
        is_valid = torch.stack((ti0, ti1, ti2, ti3), dim=-2).all(dim=-1)
        tors_mask[..., 3:7] = torch.logical_and(tors_mask[..., 3:7], is_valid)
        tors_mask[:, :, 7] = torch.logical_and(
            tors_mask[:, :, 7], mask_in[:, :, 4]
        )  # CB exist?
        tors_mask[:, :, 8] = torch.logical_and(
            tors_mask[:, :, 8], mask_in[:, :, 4]
        )  # CB exist?
        tors_mask[:, :, 9] = torch.logical_and(
            tors_mask[:, :, 9], mask_in[:, :, 5]
        )  # XG exist?

    return tors_mask


def get_torsions(
    xyz_in, seq, torsion_indices, torsion_can_flip, ref_angles, mask_in=None
):
    B, L = xyz_in.shape[:2]

    tors_mask = get_tor_mask(seq, torsion_indices, mask_in)

    # torsions to restrain to 0 or 180degree
    tors_planar = torch.zeros((B, L, 10), dtype=torch.bool, device=xyz_in.device)
    tors_planar[:, :, 5] = seq == aa2num["TYR"]  # TYR chi 3 should be planar

    # idealize given xyz coordinates before computing torsion angles
    xyz = xyz_in.clone()
    Rs, Ts = rigid_from_3_points(xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :])
    Nideal = torch.tensor([-0.5272, 1.3593, 0.000], device=xyz_in.device)
    Cideal = torch.tensor([1.5233, 0.000, 0.000], device=xyz_in.device)
    xyz[..., 0, :] = torch.einsum("brij,j->bri", Rs, Nideal) + Ts
    xyz[..., 2, :] = torch.einsum("brij,j->bri", Rs, Cideal) + Ts

    torsions = torch.zeros((B, L, 10, 2), device=xyz.device)
    # avoid undefined angles for H generation
    torsions[:, 0, 1, 0] = 1.0
    torsions[:, -1, 0, 0] = 1.0

    # omega
    torsions[:, :-1, 0, :] = th_dih(
        xyz[:, :-1, 1, :], xyz[:, :-1, 2, :], xyz[:, 1:, 0, :], xyz[:, 1:, 1, :]
    )
    # phi
    torsions[:, 1:, 1, :] = th_dih(
        xyz[:, :-1, 2, :], xyz[:, 1:, 0, :], xyz[:, 1:, 1, :], xyz[:, 1:, 2, :]
    )
    # psi
    torsions[:, :, 2, :] = -1 * th_dih(
        xyz[:, :, 0, :], xyz[:, :, 1, :], xyz[:, :, 2, :], xyz[:, :, 3, :]
    )

    # chis
    ti0 = torch.gather(xyz, 2, torsion_indices[seq, :, 0, None].repeat(1, 1, 1, 3))
    ti1 = torch.gather(xyz, 2, torsion_indices[seq, :, 1, None].repeat(1, 1, 1, 3))
    ti2 = torch.gather(xyz, 2, torsion_indices[seq, :, 2, None].repeat(1, 1, 1, 3))
    ti3 = torch.gather(xyz, 2, torsion_indices[seq, :, 3, None].repeat(1, 1, 1, 3))
    torsions[:, :, 3:7, :] = th_dih(ti0, ti1, ti2, ti3)

    # CB bend
    NC = 0.5 * (xyz[:, :, 0, :3] + xyz[:, :, 2, :3])
    CA = xyz[:, :, 1, :3]
    CB = xyz[:, :, 4, :3]
    t = th_ang_v(CB - CA, NC - CA)
    t0 = ref_angles[seq][..., 0, :]
    torsions[:, :, 7, :] = torch.stack(
        (torch.sum(t * t0, dim=-1), t[..., 0] * t0[..., 1] - t[..., 1] * t0[..., 0]),
        dim=-1,
    )

    # CB twist
    NCCA = NC - CA
    NCp = xyz[:, :, 2, :3] - xyz[:, :, 0, :3]
    NCpp = (
        NCp
        - torch.sum(NCp * NCCA, dim=-1, keepdim=True)
        / torch.sum(NCCA * NCCA, dim=-1, keepdim=True)
        * NCCA
    )
    t = th_ang_v(CB - CA, NCpp)
    t0 = ref_angles[seq][..., 1, :]
    torsions[:, :, 8, :] = torch.stack(
        (torch.sum(t * t0, dim=-1), t[..., 0] * t0[..., 1] - t[..., 1] * t0[..., 0]),
        dim=-1,
    )

    # CG bend
    CG = xyz[:, :, 5, :3]
    t = th_ang_v(CG - CB, CA - CB)
    t0 = ref_angles[seq][..., 2, :]
    torsions[:, :, 9, :] = torch.stack(
        (torch.sum(t * t0, dim=-1), t[..., 0] * t0[..., 1] - t[..., 1] * t0[..., 0]),
        dim=-1,
    )

    mask0 = torch.isnan(torsions[..., 0]).nonzero()
    mask1 = torch.isnan(torsions[..., 1]).nonzero()
    torsions[mask0[:, 0], mask0[:, 1], mask0[:, 2], 0] = 1.0
    torsions[mask1[:, 0], mask1[:, 1], mask1[:, 2], 1] = 0.0

    # alt chis
    torsions_alt = torsions.clone()
    torsions_alt[torsion_can_flip[seq, :]] *= -1

    return torsions, torsions_alt, tors_mask, tors_planar

# resolve torsion indices
@lru_cache()
def resolve_torsion_indices():
    torsion_indices = torch.full((22, 4, 4), 0)
    torsion_can_flip = torch.full((22, 10), False, dtype=torch.bool)
    for i in range(22):
        i_l, i_a = aa2long[i], aa2longalt[i]
        for j in range(4):
            if torsions[i][j] is None:
                continue
            for k in range(4):
                a = torsions[i][j][k]
                torsion_indices[i, j, k] = i_l.index(a)
                if i_l.index(a) != i_a.index(a):
                    torsion_can_flip[i, 3 + j] = True  ##bb tors never flip
    # HIS is a special case
    torsion_can_flip[8, 4] = False

    return torsion_indices, torsion_can_flip

@lru_cache()
def torsion_indices(device: torch.device = 'cpu') -> torch.LongTensor:
    indices, _ = resolve_torsion_indices()
    return indices.to(device)

@lru_cache()
def torsion_can_flip(device: torch.device = 'cpu') -> torch.BoolTensor:
    _, can_flip = resolve_torsion_indices()
    return can_flip.to(device)

# kinematic parameters
@lru_cache()
def kinematic_parameters():
    _torsion_indices = torsion_indices()

    base_indices = torch.full((22, 27), 0, dtype=torch.long)
    xyzs_in_base_frame = torch.ones((22, 27, 4))
    RTs_by_torsion = torch.eye(4).repeat(22, 7, 1, 1)
    reference_angles = torch.ones((22, 3, 2))

    for i in range(22):
        i_l = aa2long[i]
        for name, base, coords in ideal_coords[i]:
            idx = i_l.index(name)
            base_indices[i, idx] = base
            xyzs_in_base_frame[i, idx, :3] = torch.tensor(coords)

        # omega frame
        RTs_by_torsion[i, 0, :3, :3] = torch.eye(3)
        RTs_by_torsion[i, 0, :3, 3] = torch.zeros(3)

        # phi frame
        RTs_by_torsion[i, 1, :3, :3] = make_frame(
            xyzs_in_base_frame[i, 0, :3] - xyzs_in_base_frame[i, 1, :3],
            torch.tensor([1.0, 0.0, 0.0]),
        )
        RTs_by_torsion[i, 1, :3, 3] = xyzs_in_base_frame[i, 0, :3]

        # psi frame
        RTs_by_torsion[i, 2, :3, :3] = make_frame(
            xyzs_in_base_frame[i, 2, :3] - xyzs_in_base_frame[i, 1, :3],
            xyzs_in_base_frame[i, 1, :3] - xyzs_in_base_frame[i, 0, :3],
        )
        RTs_by_torsion[i, 2, :3, 3] = xyzs_in_base_frame[i, 2, :3]

        # chi1 frame
        if torsions[i][0] is not None:
            a0, a1, a2 = _torsion_indices[i, 0, 0:3]
            RTs_by_torsion[i, 3, :3, :3] = make_frame(
                xyzs_in_base_frame[i, a2, :3] - xyzs_in_base_frame[i, a1, :3],
                xyzs_in_base_frame[i, a0, :3] - xyzs_in_base_frame[i, a1, :3],
            )
            RTs_by_torsion[i, 3, :3, 3] = xyzs_in_base_frame[i, a2, :3]

        # chi2~4 frame
        for j in range(1, 4):
            if torsions[i][j] is not None:
                a2 = _torsion_indices[i, j, 2]
                if (i == 18 and j == 2) or (
                    i == 8 and j == 2
                ):  # TYR CZ-OH & HIS CE1-HE1 a special case
                    a0, a1 = _torsion_indices[i, j, 0:2]
                    RTs_by_torsion[i, 3 + j, :3, :3] = make_frame(
                        xyzs_in_base_frame[i, a2, :3] - xyzs_in_base_frame[i, a1, :3],
                        xyzs_in_base_frame[i, a0, :3] - xyzs_in_base_frame[i, a1, :3],
                    )
                else:
                    RTs_by_torsion[i, 3 + j, :3, :3] = make_frame(
                        xyzs_in_base_frame[i, a2, :3],
                        torch.tensor([-1.0, 0.0, 0.0]),
                    )
                RTs_by_torsion[i, 3 + j, :3, 3] = xyzs_in_base_frame[i, a2, :3]

        # CB/CG angles
        NCr = 0.5 * (xyzs_in_base_frame[i, 0, :3] + xyzs_in_base_frame[i, 2, :3])
        CAr = xyzs_in_base_frame[i, 1, :3]
        CBr = xyzs_in_base_frame[i, 4, :3]
        CGr = xyzs_in_base_frame[i, 5, :3]
        reference_angles[i, 0, :] = th_ang_v(CBr - CAr, NCr - CAr)
        NCp = xyzs_in_base_frame[i, 2, :3] - xyzs_in_base_frame[i, 0, :3]
        NCpp = NCp - torch.dot(NCp, NCr) / torch.dot(NCr, NCr) * NCr
        reference_angles[i, 1, :] = th_ang_v(CBr - CAr, NCpp)
        reference_angles[i, 2, :] = th_ang_v(CGr, torch.tensor([-1.0, 0.0, 0.0]))

    return base_indices, xyzs_in_base_frame, RTs_by_torsion, reference_angles

@lru_cache()
def base_indices(device: torch.device = 'cpu') -> torch.LongTensor:
    indices,*_ = kinematic_parameters()
    return indices.to(device)

@lru_cache()
def xyzs_in_base_frame(device: torch.device = 'cpu') -> torch.Tensor:
    _, xyzs,_,_ = kinematic_parameters()
    return xyzs.to(device)

@lru_cache()
def RTs_by_torsion(device: torch.device = 'cpu') -> torch.Tensor:
    _,_, rts, _ = kinematic_parameters()
    return rts.to(device)

@lru_cache()
def reference_angles(device: torch.device = 'cpu') -> torch.Tensor:
    *_, angles = kinematic_parameters()
    return angles.to(device)

def get_torsions_initialized(   xyz_in: torch.Tensor, 
                                seq: torch.Tensor, 
                                mask_in: torch.BoolTensor | None = None) -> tuple[torch.Tensor,torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Get torsions with initialized residue properties'''
    _torsion_indices = torsion_indices(xyz_in.device)
    _torsion_can_flip = torsion_can_flip(xyz_in.device)
    _reference_angles = reference_angles(xyz_in.device)
    return get_torsions(xyz_in, seq, _torsion_indices, _torsion_can_flip, _reference_angles, mask_in)
