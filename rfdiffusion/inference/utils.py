
from typing import Literal, Callable, Any
import random
import logging
import glob
import os

import numpy as np
from omegaconf import DictConfig
import torch
from torch import Tensor, BoolTensor, LongTensor
from quatorch import Quaternion

from rfdiffusion.potentials.manager import PotentialManager
from rfdiffusion.diffusion import get_beta_schedule, Diffuser
from rfdiffusion.util import rigid_from_3_points
from rfdiffusion import util

###########################################################
#### Functions which can be called outside of Denoiser ####
###########################################################


def get_next_frames(
    xt: Tensor, 
    px0: Tensor, 
    t: Tensor, 
    diffuser: Diffuser, 
    so3_type: Literal['igso3'], 
    diffusion_mask: BoolTensor, 
    noise_scale: float = 1.0
    ) -> Tensor:
    """
    get_next_frames gets updated frames using IGSO(3) + score_based reverse diffusion.


    based on self.so3_type use score based update.

    Generate frames at t-1
    Rather than generating random rotations (as occurs during forward process), calculate rotation between xt and px0

    Args:
        xt: noised coordinates of shape [L, 14, 3]
        px0: prediction of coordinates at t=0, of shape [L, 14, 3]
        t: integer time step
        diffuser: Diffuser object for reverse igSO3 sampling
        so3_type: The type of SO3 noising being used ('igso3')
        diffusion_mask: of shape [L] of type bool, True means not to be
            updated (e.g. mask is true for motif residues)
        noise_scale: scale factor for the noise added (IGSO3 only)

    Returns:
        backbone coordinates for step x_t-1 of shape [L, 3, 3]
    """
    N_0 = px0[None, :, 0, :]
    Ca_0 = px0[None, :, 1, :]
    C_0 = px0[None, :, 2, :]

    R_0, Ca_0 = rigid_from_3_points(N_0, Ca_0, C_0)

    N_t = xt[None, :, 0, :]
    Ca_t = xt[None, :, 1, :]
    C_t = xt[None, :, 2, :]

    R_t, Ca_t = rigid_from_3_points(N_t, Ca_t, C_t)

    # this must be to normalize them or something
    R_0 = Quaternion.from_rotation_matrix(R_0.squeeze())
    R_t = Quaternion.from_rotation_matrix(R_t.squeeze())

    all_rot_transitions = Quaternion(R_t.new_ones(R_t.shape[:-1]), *R_t.new_zeros((3,)+R_t.shape[:-1]))
    # Sample next frame for each residue
    if so3_type == "igso3":
        # don't do calculations on masked positions since they end up as identity matrix
        all_rot_transitions[~diffusion_mask] = diffuser.so3_diffuser.reverse_sample_vectorized(
            R_t[~diffusion_mask].as_subclass(Quaternion),
            R_0[~diffusion_mask].as_subclass(Quaternion),
            t,
            noise_level=noise_scale,
            mask=None,
            return_perturb=True,
        )
    else:
        raise ValueError(f"so3 diffusion type {so3_type} not implemented")

    # Apply the interpolated rotation matrices to the coordinates
    next_crds = all_rot_transitions.unsqueeze(1).rotate_vector(
            (xt[:, :3, :] - Ca_t.squeeze()[:, None])
        ) + Ca_t.squeeze()[:, None]

    # (L,3,3) set of backbone coordinates with slight rotation
    return next_crds


def get_mu_xt_x0(
    xt: Tensor, 
    px0: Tensor, 
    t: LongTensor, 
    beta_schedule: Tensor, 
    alphabar_schedule: Tensor, 
    eps: float = 1e-6
    ) -> tuple[Tensor,Tensor]:
    """
    Given xt, predicted x0 and the timestep t, give mu of x(t-1)
    Assumes t is 0 indexed
    """
    # sigma is predefined from beta. Often referred to as beta tilde t
    t_idx = t - 1
    sigma = (
        (1 - alphabar_schedule[t_idx - 1]) / (1 - alphabar_schedule[t_idx])
    ) * beta_schedule[t_idx]

    xt_ca = xt[:, 1, :]
    px0_ca = px0[:, 1, :]

    a = (
        (torch.sqrt(alphabar_schedule[t_idx - 1] + eps) * beta_schedule[t_idx])
        / (1 - alphabar_schedule[t_idx])
    ) * px0_ca
    b = (
        (
            torch.sqrt(1 - beta_schedule[t_idx] + eps)
            * (1 - alphabar_schedule[t_idx - 1])
        )
        / (1 - alphabar_schedule[t_idx])
    ) * xt_ca

    mu = a + b

    return mu, sigma


def get_next_ca(
    xt: Tensor,
    px0: Tensor,
    t: LongTensor,
    diffusion_mask: BoolTensor | None,
    crd_scale : float,
    beta_schedule,
    alphabar_schedule,
    noise_scale: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
    """
    Given full atom x0 prediction (xyz coordinates), diffuse to x(t-1)

    Parameters:

        xt (L, 14/27, 3) set of coordinates

        px0 (L, 14/27, 3) set of coordinates

        t: time step. Note this is zero-index current time step, so are generating t-1

        logits_aa (L x 20 ) amino acid probabilities at each position

        seq_schedule (L): Tensor of bools, True is unmasked, False is masked. For this specific t

        diffusion_mask (torch.tensor, required): Tensor of bools, True means NOT diffused at this residue, False means diffused

        noise_scale: scale factor for the noise being added

    """
    # bring to origin after global alignment (when don't have a motif) or replace input motif and bring to origin, and then scale
    px0 = px0 * crd_scale
    xt = xt * crd_scale

    # get mu(xt, x0)
    mu, sigma = get_mu_xt_x0(
        xt, px0, t, beta_schedule=beta_schedule, alphabar_schedule=alphabar_schedule
    )

    sampled_crds = torch.normal(mu, torch.sqrt(sigma * noise_scale))
    delta = sampled_crds - xt[:, 1, :]  # check sign of this is correct

    if not diffusion_mask is None:
        # Don't move motif
        delta[diffusion_mask, ...] = 0

    out_crds = xt + delta[:, None, :]

    return out_crds / crd_scale, delta / crd_scale


def get_noise_schedule(T: int, noiseT: float, noise1: float, schedule_type: Literal['constant','linear']) -> Callable[[int | LongTensor],float | Tensor]:
    """
    Function to create a schedule that varies the scale of noise given to the model over time

    Parameters:

        T: The total number of timesteps in the denoising trajectory

        noiseT: The inital (t=T) noise scale

        noise1: The final (t=1) noise scale

        schedule_type: The type of function to use to interpolate between noiseT and noise1

    Returns:

        noise_schedule: A function which maps timestep to noise scale

    """

    match schedule_type:
        case 'constant':
            return lambda t: noiseT
        case "linear": 
            return lambda t: ((t - 1) / (T - 1)) * (noiseT - noise1) + noise1
        case _:
            raise ValueError(f"noise_schedule must be one of ['constant','linear']. Received noise_schedule={schedule_type}. Exiting.")


class Denoise:
    """
    Class for getting x(t-1) from predicted x0 and x(t)
    Strategy:
        Ca coordinates: Rediffuse to x(t-1) from predicted x0
        Frames: Approximate update from rotation score
        Torsions: 1/t of the way to the x0 prediction

    """

    def __init__(
        self,
        T: int,
        L: int,
        diffuser: Diffuser,
        b_0: float = 0.001,
        b_T: float = 0.1,
        min_b: float = 1.0,
        max_b: float = 12.5,
        min_sigma: float = 0.05,
        max_sigma: float = 1.5,
        noise_level: float = 0.5,
        schedule_type: Literal['constant','linear'] = "linear",
        so3_schedule_type: Literal['constant','linear'] = "linear",
        schedule_kwargs: dict = {},
        so3_type: Literal ['igso3'] = "igso3",
        noise_scale_ca: float = 1.0,
        final_noise_scale_ca: float = 1.0,
        ca_noise_schedule_type: Literal['constant','linear'] = "constant",
        noise_scale_frame: float = 0.5,
        final_noise_scale_frame: float = 0.5,
        frame_noise_schedule_type: Literal['constant','linear'] = "constant",
        crd_scale: float = 1 / 15,
        potential_manager: PotentialManager | None =None,
        partial_T=None,
    ):
        """

        Parameters:
            noise_level: scaling on the noise added (set to 0 to use no noise,
                to 1 to have full noise)

        """
        self.T = T
        self.L = L
        self.diffuser = diffuser
        self.b_0 = b_0
        self.b_T = b_T
        self.noise_level = noise_level
        self.schedule_type = schedule_type
        self.so3_type = so3_type
        self.crd_scale = crd_scale
        self.noise_scale_ca = noise_scale_ca
        self.final_noise_scale_ca = final_noise_scale_ca
        self.ca_noise_schedule_type = ca_noise_schedule_type
        self.noise_scale_frame = noise_scale_frame
        self.final_noise_scale_frame = final_noise_scale_frame
        self.frame_noise_schedule_type = frame_noise_schedule_type
        self.potential_manager = potential_manager
        self._log = logging.getLogger(__name__)

        self.schedule, self.alpha_schedule, self.alphabar_schedule = get_beta_schedule(
            self.T, self.b_0, self.b_T, self.schedule_type, inference=True
        )

        self.noise_schedule_ca = get_noise_schedule(
            self.T,
            self.noise_scale_ca,
            self.final_noise_scale_ca,
            self.ca_noise_schedule_type,
        )
        self.noise_schedule_frame = get_noise_schedule(
            self.T,
            self.noise_scale_frame,
            self.final_noise_scale_frame,
            self.frame_noise_schedule_type,
        )

    def align_to_xt_motif(self, px0: Tensor, xT: Tensor, diffusion_mask: BoolTensor, eps: float = 1e-6) -> Tensor:
        """
        Need to align px0 to motif in xT. This is to permit the swapping of residue positions in the px0 motif for the true coordinates.
        First, get rotation matrix from px0 to xT for the motif residues.
        Second, rotate px0 (whole structure) by that rotation matrix
        Third, centre at origin
        """

        def rmsd(V: Tensor, W: Tensor, eps: float = 0.0) -> Tensor:
            # First sum down atoms, then sum down xyz
            N = V.shape[-2]
            return torch.sqrt((V - W).square().sum(dim=(-2,-1)) / N + eps)

        assert (
            xT.shape[1] == px0.shape[1]
        ), f"xT has shape {xT.shape} and px0 has shape {px0.shape}"

        L, n_atom, _ = xT.shape  # A is number of atoms
        atom_mask = ~torch.isnan(px0)

        # 1 centre motifs at origin and get rotation matrix
        px0_motif = px0[diffusion_mask, :3].reshape(-1, 3)
        xT_motif = xT[diffusion_mask, :3].reshape(-1, 3)
        px0_motif_mean = px0_motif.mean(0)  # need later
        xT_motif_mean = xT_motif.mean(0)

        # center at origin
        px0_motif = px0_motif - px0_motif_mean
        xT_motif = xT_motif - xT_motif_mean

        # compute optimal rotation matrix using SVD
        U, S, Vt = torch.svd(xT_motif.T @ px0_motif)

        # construct rotation matrix
        R = U @ Vt

        # get rotated coords
        rB = px0_motif @ R.T

        # calculate rmsd
        rms = rmsd(xT_motif, rB)
        self._log.info(f"Sampled motif RMSD: {rms:.2f}")

        # 2 rotate whole px0 by rotation matrix
        px0[~atom_mask] = 0  # convert nans to 0
        px0 = px0.reshape(-1, 3) - px0_motif_mean
        px0_ = px0 @ R.T

        # 3 put in same global position as xT
        px0_ = px0_ + xT_motif_mean
        px0_ = px0_.reshape([L, n_atom, 3])
        px0_[~atom_mask] = float("nan")
        return px0_

    def get_potential_gradients(self, xyz: Tensor, diffusion_mask: BoolTensor) -> Tensor:
        """
        This could be moved into potential manager if desired - NRB

        Function to take a structure (x) and get per-atom gradients used to guide diffusion update

        Inputs:

            xyz (torch.tensor, required): [L,27,3] Coordinates at which the gradient will be computed

        Outputs:

            Ca_grads (torch.tensor): [L,3] The gradient at each Ca atom
        """

        if self.potential_manager is None or self.potential_manager.is_empty():
            return xyz.new_zeros(xyz.shape[0], 3)

        xyz.requires_grad = True

        if not xyz.grad is None:
            xyz.grad.zero_()

        current_potential = self.potential_manager.compute_all_potentials(xyz)
        current_potential.backward()

        # Since we are not moving frames, Cb grads are same as Ca grads
        # Need access to calculated Cb coordinates to be able to get Cb grads though
        Ca_grads = xyz.grad[:, 1, :]

        if diffusion_mask is not None:
            Ca_grads[diffusion_mask, :] = 0.0

        # check for NaN's
        if torch.isnan(Ca_grads).any():
            print("WARNING: NaN in potential gradients, replacing with zero grad.")
            Ca_grads[:] = 0.0

        return Ca_grads

    def get_next_pose(
        self,
        xt: Tensor,
        px0: Tensor,
        t: LongTensor,
        diffusion_mask: BoolTensor,
        fix_motif: bool = True,
        align_motif: bool = True,
        include_motif_sidechains: bool = True,
    ) -> tuple[Tensor,Tensor]:
        """
        Wrapper function to take px0, xt and t, and to produce xt-1
        First, aligns px0 to xt
        Then gets coordinates, frames and torsion angles

        Parameters:

            xt (torch.tensor, required): Current coordinates at timestep t

            px0 (torch.tensor, required): Prediction of x0

            t (int, required): timestep t

            diffusion_mask (torch.tensor, required): Mask for structure diffusion

            fix_motif (bool): Fix the motif structure

            align_motif (bool): Align the model's prediction of the motif to the input motif

            include_motif_sidechains (bool): Provide sidechains of the fixed motif to the model
        """

        assert (xt.shape[1] == 14) or (xt.shape[1] == 27)
        assert (px0.shape[1] == 14) or (px0.shape[1] == 27)

        ###############################
        ### Align pX0 onto Xt motif ###
        ###############################

        if align_motif and diffusion_mask.any():
            px0 = self.align_to_xt_motif(px0, xt, diffusion_mask)
        # xT_motif_aligned = self.align_to_xt_motif(px0, xt, diffusion_mask)

        px0 = px0.to(xt.device)
        # Now done with diffusion mask. if fix motif is False, just set diffusion mask to be all True, and all coordinates can diffuse
        if not fix_motif:
            diffusion_mask[:] = False

        # get the next set of CA coordinates
        noise_scale_ca = self.noise_schedule_ca(t)
        _, ca_deltas = get_next_ca(
            xt,
            px0,
            t,
            diffusion_mask,
            crd_scale=self.crd_scale,
            beta_schedule=self.schedule,
            alphabar_schedule=self.alphabar_schedule,
            noise_scale=noise_scale_ca,
        )

        # get the next set of backbone frames (coordinates)
        noise_scale_frame = self.noise_schedule_frame(t)
        frames_next = get_next_frames(
            xt,
            px0,
            t,
            diffuser=self.diffuser,
            so3_type=self.so3_type,
            diffusion_mask=diffusion_mask,
            noise_scale=noise_scale_frame,
        )

        # Apply gradient step from guiding potentials
        # This can be moved to below where the full atom representation is calculated to allow for potentials involving sidechains

        grad_ca = self.get_potential_gradients(
            xt.clone(), diffusion_mask=diffusion_mask
        )

        ca_deltas += self.potential_manager.get_guide_scale(t) * grad_ca

        # add the delta to the new frames
        frames_next = frames_next + ca_deltas[:, None, :]  # translate

        fullatom_next = torch.full_like(xt, float("nan")).unsqueeze(0)
        fullatom_next[:, :, :3] = frames_next[None]

        if include_motif_sidechains:
            fullatom_next[:, diffusion_mask, :14] = xt[None, diffusion_mask]

        return fullatom_next.squeeze()[:, :14, :], px0


def parse_pdb(filename: str, **kwargs) -> dict[str, Any]:
    """extract xyz coords for all heavy atoms"""
    with open(filename,"r") as f:
        lines=f.readlines()
    return parse_pdb_lines(lines, **kwargs)


def parse_pdb_lines(lines: list[str], parse_hetatom: bool = False, ignore_het_h: bool = True) -> dict[str, Any]:
    # indices of residues observed in the structure
    res, pdb_idx = [],[]
    for l in lines:
        if l[:4] == "ATOM" and l[12:16].strip() == "CA":
            res.append((l[22:26], l[17:20]))
            # chain letter, res num
            pdb_idx.append((l[21:22].strip(), int(l[22:26].strip())))
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
    pdb_idx = [
        (l[21:22].strip(), int(l[22:26].strip()))
        for l in lines
        if l[:4] == "ATOM" and l[12:16].strip() == "CA"
    ]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = (
            l[21:22],
            int(l[22:26]),
            " " + l[12:16].strip().ljust(3),
            l[17:20],
        )
        if (chain,resNo) in pdb_idx:
            idx = pdb_idx.index((chain, resNo))
            # for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            for i_atm, tgtatm in enumerate(
                util.aa2long[util.aa2num[aa]][:14]
                ):
                if (
                    tgtatm is not None and tgtatm.strip() == atom.strip()
                    ):  # ignore whitespace
                    xyz[idx, i_atm, :] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz[..., 0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i, idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]

    seq = np.array(seq)[i_unique]

    out = {
        "xyz": xyz,  # cartesian coordinates, [Lx14]
        "mask": mask,  # mask showing which atoms are present in the PDB file, [Lx14]
        "idx": np.array(
            [i[1] for i in pdb_idx]
        ),  # residue numbers in the PDB file, [L]
        "seq": np.array(seq),  # amino acid sequence, [L]
        "pdb_idx": pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
    }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6] == "HETATM" and not (ignore_het_h and l[77] == "H"):
                info_het.append(
                    dict(
                        idx=int(l[7:11]),
                        atom_id=l[12:16],
                        atom_type=l[77],
                        name=l[16:20],
                    )
                )
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out["xyz_het"] = np.array(xyz_het)
        out["info_het"] = info_het

    return out


def process_target(pdb_path: str, parse_hetatom: bool = False, center: bool = True) -> dict[str, Any]:
    # Read target pdb and extract features.
    target_struct = parse_pdb(pdb_path, parse_hetatom=parse_hetatom)

    # Zero-center positions
    ca_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True)
    if not center:
        ca_center = 0
    xyz = torch.from_numpy(target_struct["xyz"] - ca_center)
    seq_orig = torch.from_numpy(target_struct["seq"])
    atom_mask = torch.from_numpy(target_struct["mask"])
    seq_len = len(xyz)

    # Make 27 atom representation
    xyz_27 = torch.full((seq_len, 27, 3), np.nan).float()
    xyz_27[:, :14, :] = xyz[:, :14, :]

    mask_27 = torch.full((seq_len, 27), False)
    mask_27[:, :14] = atom_mask
    out = {
        "xyz_27": xyz_27,
        "mask_27": mask_27,
        "seq": seq_orig,
        "pdb_idx": target_struct["pdb_idx"],
    }
    if parse_hetatom:
        out["xyz_het"] = target_struct["xyz_het"]
        out["info_het"] = target_struct["info_het"]
    return out


def get_idx0_hotspots(mappings: dict, ppi_conf: DictConfig, binderlen: int) -> list[int]:
    """
    Take pdb-indexed hotspot resudes and the length of the binder, and makes the 0-indexed tensor of hotspots
    """

    if binderlen == 0 or ppi_conf.hotspot_res is None:
        return None

    assert all(
        [i[0].isalpha() for i in ppi_conf.hotspot_res]
    ), "Hotspot residues need to be provided in pdb-indexed form. E.g. A100,A103"

    hotspots = [(i[0], int(i[1:])) for i in ppi_conf.hotspot_res]
    hotspot_idx = []
    for i, res in enumerate(mappings["receptor_con_ref_pdb_idx"]):
        if res in hotspots:
            hotspot_idx.append(mappings["receptor_con_hal_idx0"][i])
    
    return hotspot_idx


class BlockAdjacency:
    """
    Class for handling PPI design inference with ss/block_adj inputs.
    Basic idea is to provide a list of scaffolds, and to output ss and adjacency
    matrices based off of these, while sampling additional lengths.
    Inputs:
        - scaffold_list: list of scaffolds (e.g. ['2kl8','1cif']). Can also be a .txt file.
        - scaffold dir: directory where scaffold ss and adj are precalculated
        - sampled_insertion: how many additional residues do you want to add to each loop segment? Randomly sampled 0-this number (or within given range)
        - sampled_N: randomly sample up to this number of additional residues at N-term
        - sampled_C: randomly sample up to this number of additional residues at C-term
        - ss_mask: how many residues do you want to mask at either end of a ss (H or E) block. Fixed value
        - num_designs: how many designs are you wanting to generate? Currently only used for bookkeeping
        - systematic: do you want to systematically work through the list of scaffolds, or randomly sample (default)
        - num_designs_per_input: Not really implemented yet. Maybe not necessary
    Outputs:
        - L: new length of chain to be diffused
        - ss: all loops and insertions, and ends of ss blocks (up to ss_mask) set to mask token (3). Onehot encoded. (L,4)
        - adj: block adjacency with equivalent masking as ss (L,L)
    """

    def __init__(self, conf: DictConfig, num_designs: int):
        """
        Parameters:
          inputs:
             conf.scaffold_list as conf
             conf.inference.num_designs for sanity checking
        """
       
        self.conf=conf 
        # either list or path to .txt file with list of scaffolds
        match self.conf.scaffoldguided.scaffold_list:
            case list() as scaffold_list:
                self.scaffold_list = scaffold_list
            case str() as scaffold_file if scaffold_file[-4:] == ".txt":
                # txt file with list of ids
                with open(scaffold_file, "r") as f:
                    self.scaffold_list = [line.strip() for line in f.readlines()]
            case None:
                self.scaffold_list = [
                    os.path.split(i)[1][:-6]
                    for i in glob.glob(f"{self.conf.scaffoldguided.scaffold_dir}/*_ss.pt")
                ]
                self.scaffold_list.sort()
            case _:
                raise NotImplementedError

        # path to directory with scaffolds, ss files and block_adjacency files
        self.scaffold_dir = self.conf.scaffoldguided.scaffold_dir

        # maximum sampled insertion in each loop segment
        if "-" in str(self.conf.scaffoldguided.sampled_insertion):
            self.sampled_insertion = [
                int(str(self.conf.scaffoldguided.sampled_insertion).split("-")[0]),
                int(str(self.conf.scaffoldguided.sampled_insertion).split("-")[1]),
            ]
        else:
            self.sampled_insertion = [0, int(self.conf.scaffoldguided.sampled_insertion)]

        # maximum sampled insertion at N- and C-terminus
        if "-" in str(self.conf.scaffoldguided.sampled_N):
            self.sampled_N = [
                int(str(self.conf.scaffoldguided.sampled_N).split("-")[0]),
                int(str(self.conf.scaffoldguided.sampled_N).split("-")[1]),
            ]
        else:
            self.sampled_N = [0, int(self.conf.scaffoldguided.sampled_N)]
        if "-" in str(self.conf.scaffoldguided.sampled_C):
            self.sampled_C = [
                int(str(self.conf.scaffoldguided.sampled_C).split("-")[0]),
                int(str(self.conf.scaffoldguided.sampled_C).split("-")[1]),
            ]
        else:
            self.sampled_C = [0, int(self.conf.scaffoldguided.sampled_C)]

        # number of residues to mask ss identity of in H/E regions (from junction)
        # e.g. if ss_mask = 2, L,L,L,H,H,H,H,H,H,H,L,L,E,E,E,E,E,E,L,L,L,L,L,L would become\
        # M,M,M,M,M,H,H,H,M,M,M,M,M,M,E,E,M,M,M,M,M,M,M,M where M is mask
        self.ss_mask = self.conf.scaffoldguided.ss_mask

        # whether or not to work systematically through the list
        self.systematic = self.conf.scaffoldguided.systematic

        self.num_designs = num_designs

        if len(self.scaffold_list) > self.num_designs:
            print(
                "WARNING: Scaffold set is bigger than num_designs, so not every scaffold type will be sampled"
            )

        # for tracking number of designs
        self.num_completed = 0
        if self.systematic:
            self.item_n = 0

        # whether to mask loops or not
        self.mask_loops = self.conf.scaffoldguided.mask_loops
        if not self.mask_loops:
            assert self.conf.scaffoldguided.sampled_N == 0, "can't add length if not masking loops"
            assert self.conf.scaffoldguided.sampled_C == 0, "can't add lemgth if not masking loops"
            assert self.conf.scaffoldguided.sampled_insertion == 0, "can't add length if not masking loops"

    def get_ss_adj(self, item: str) -> tuple[Tensor, Tensor]:
        """
        Given at item, get the ss tensor and block adjacency matrix for that item
        """
        ss = torch.load(os.path.join(self.scaffold_dir, f'{item.split(".")[0]}_ss.pt'))
        adj = torch.load(
            os.path.join(self.scaffold_dir, f'{item.split(".")[0]}_adj.pt')
        )

        return ss, adj

    def mask_to_segments(self, mask: BoolTensor) -> list[tuple[Literal['ss','loop'],int]]:
        """
        Takes a mask of True (loop) and False (non-loop), and outputs list of tuples (loop or not, length of element)
        """
        jumps = mask.diff()
        begin = torch.nonzero(jumps).squeeze(-1) # non-trivial segment starts
        lengths = begin.diff(prepend = torch.tensor([-1]), append = torch.tensor([len(mask)-1]))
        values = torch.concat([mask[:1],mask[begin]])
        return [("loop" if val.item() else "ss", length.item()) for val, length in zip(values, lengths)]

    def expand_mask(self, mask: BoolTensor, segments: list[tuple[Literal['ss','loop'],int]]) -> Tensor:
        """
        Function to generate a new mask with dilated loops and N and C terminal additions
        """
        N_add = random.randint(self.sampled_N[0], self.sampled_N[1])
        C_add = random.randint(self.sampled_C[0], self.sampled_C[1])

        output = N_add * [False]
        for ss, length in segments:
            if ss == "ss":
                output.extend(length * [True])
            else:
                # randomly sample insertion length
                ins = random.randint(
                    self.sampled_insertion[0], self.sampled_insertion[1]
                )
                output.extend((length + ins) * [False])
        output.extend(C_add * [False])
        assert torch.sum(torch.tensor(output)) == torch.sum(~mask)
        return torch.tensor(output)

    def expand_ss(
        self, 
        ss: Tensor, 
        adj: Tensor, 
        mask: BoolTensor, 
        expanded_mask: BoolTensor
        ) -> tuple[Tensor, Tensor]:
        """
        Given an expanded mask, populate a new ss and adj based on this
        """
        ss_out = torch.ones(expanded_mask.shape[0]) * 3  # set to mask token
        adj_out = torch.full((expanded_mask.shape[0], expanded_mask.shape[0]), 0.0)
        ss_out[expanded_mask] = ss[~mask]
        expanded_mask_2d = torch.full(adj_out.shape, True)
        # mask out loops/insertions, which is ~expanded_mask
        expanded_mask_2d[~expanded_mask, :] = False
        expanded_mask_2d[:, ~expanded_mask] = False

        mask_2d = torch.full(adj.shape, True)
        # mask out loops. This mask is True=loop
        mask_2d[mask, :] = False
        mask_2d[:, mask] = False
        adj_out[expanded_mask_2d] = adj[mask_2d]
        adj_out = adj_out.reshape((expanded_mask.shape[0], expanded_mask.shape[0]))

        return ss_out, adj_out

    def mask_ss_adj(self, ss: Tensor, adj: Tensor, expanded_mask: BoolTensor) -> tuple[Tensor, Tensor]:
        """
        Given an expanded ss and adj, mask some number of residues at either end of non-loop ss
        """
        original_mask = torch.clone(expanded_mask)
        if self.ss_mask > 0:
            for i in range(1, self.ss_mask + 1):
                expanded_mask[i:] *= original_mask[:-i]
                expanded_mask[:-i] *= original_mask[i:]

        if self.mask_loops:
            ss[~expanded_mask] = 3
            adj[~expanded_mask, :] = 0
            adj[:, ~expanded_mask] = 0

        # mask adjacency
        adj[~expanded_mask] = 2
        adj[:, ~expanded_mask] = 2

        return ss, adj

    def get_scaffold(self) -> tuple[int, Tensor]:
        """
        Wrapper method for pulling an item from the list, and preparing ss and block adj features
        """
        
        # Handle determinism. Useful for integration tests
        if self.conf.inference.deterministic:
            torch.manual_seed(self.num_completed)
            np.random.seed(self.num_completed)
            random.seed(self.num_completed)
  
        if self.systematic:
            # reset if num designs > num_scaffolds
            if self.item_n >= len(self.scaffold_list):
                self.item_n = 0
            item = self.scaffold_list[self.item_n]
            self.item_n += 1
        else:
            item = random.choice(self.scaffold_list)
        print("Scaffold constrained based on file: ", item)
        # load files
        ss, adj = self.get_ss_adj(item)
        adj_orig = torch.clone(adj)
        # separate into segments (loop or not)
        mask = torch.where(ss == 2, 1, 0).bool()
        segments = self.mask_to_segments(mask)

        # insert into loops to generate new mask
        expanded_mask = self.expand_mask(mask, segments)

        # expand ss and adj
        ss, adj = self.expand_ss(ss, adj, mask, expanded_mask)

        # finally, mask some proportion of the ss at either end of the non-loop ss blocks
        ss, adj = self.mask_ss_adj(ss, adj, expanded_mask)

        # and then update num_completed
        self.num_completed += 1

        return ss.shape[0], torch.nn.functional.one_hot(ss.long(), num_classes=4), adj


class Target:
    """
    Class to handle targets (fixed chains).
    Inputs:
        - path to pdb file
        - hotspot residues, in the form B10,B12,B60 etc
        - whether or not to crop, and with which method
    Outputs:
        - Dictionary of xyz coordinates, indices, pdb_indices, pdb mask
    """

    def __init__(self, conf: DictConfig, hotspots: list | None = None):
        self.pdb = parse_pdb(conf.target_path)

        if hotspots is not None:
            self.hotspots = hotspots
        else:
            self.hotspots = []
        self.pdb["hotspots"] = np.array(
            [
                True if f"{i[0]}{i[1]}" in self.hotspots else False
                for i in self.pdb["pdb_idx"]
            ]
        )

        if conf.contig_crop:
            self.contig_crop(conf.contig_crop)

    @classmethod
    def parse_contig(cls, contig_crop):
        """
        Takes contig input and parses
        """
        contig_list = []
        for contig in contig_crop[0].split(" "):
            subcon = []
            for crop in contig.split("/"):
                if crop[0].isalpha():
                    subcon.extend(
                        [
                            (crop[0], p)
                            for p in np.arange(
                                int(crop.split("-")[0][1:]), int(crop.split("-")[1]) + 1
                            )
                        ]
                    )
            contig_list.append(subcon)
        return contig_list

    def contig_crop(self, contig_crop, residue_offset=200) -> None:
        """
        Method to take a contig string referring to the receptor and output a pdb dictionary with just this crop
        NB there are two ways to provide inputs:
            - 1) e.g. B1-30,0 B50-60,0. This will add a residue offset between each chunk
            - 2) e.g. B1-30,B50-60,B80-100. This will keep the original indexing of the pdb file.
        Can handle the target being on multiple chains
        """

        # add residue offset between chains if multiple chains in receptor file
        for idx, val in enumerate(self.pdb["pdb_idx"]):
            if idx != 0 and val != self.pdb["pdb_idx"][idx - 1]:
                self.pdb["idx"][idx:] += residue_offset + idx

        # convert contig to mask
        contig_list = self.parse_contig(contig_crop)

        # add residue offset to different parts of contig_list
        for contig in contig_list[1:]:
            start = int(contig[0][1])
            self.pdb["idx"][start:] += residue_offset
        # flatten list
        contig_list = [i for j in contig_list for i in j]
        mask = np.array(
            [True if i in contig_list else False for i in self.pdb["pdb_idx"]]
        )

        # sanity check
        assert np.sum(self.pdb["hotspots"]) == np.sum(
            self.pdb["hotspots"][mask]
        ), "Supplied hotspot residues are missing from the target contig!"
        # crop pdb
        for key, val in self.pdb.items():
            try:
                self.pdb[key] = val[mask]
            except:
                self.pdb[key] = [i for idx, i in enumerate(val) if mask[idx]]
        self.pdb["crop_mask"] = mask

    def get_target(self) -> dict:
        return self.pdb

def ss_from_contig(ss_masks: dict) -> LongTensor:
    """  
    Function for taking 1D masks for each of the ss types, and outputting a secondary structure input
    """
    L=len(ss_masks['helix'])
    ss=torch.zeros((L, 4)).long()
    ss[:,3] = 1 #mask
    for idx, mask in enumerate([ss_masks['helix'],ss_masks['strand'], ss_masks['loop']]):
        ss[mask,idx] = 1
        ss[mask, 3] = 0 # remove the mask token
    return ss
