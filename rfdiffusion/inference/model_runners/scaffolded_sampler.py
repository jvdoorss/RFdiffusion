import torch
from torch import Tensor, LongTensor
import numpy as np
import torch.nn.functional as nn
from omegaconf import DictConfig, OmegaConf

from rfdiffusion.kinematics import get_init_xyz, xyz_to_t2d
from rfdiffusion.chemical import seq2chars
from rfdiffusion.contigs import ContigMap
from rfdiffusion.potentials.manager import PotentialManager

from rfdiffusion.inference.utils import Target, BlockAdjacency, process_target, get_idx0_hotspots, ss_from_contig

from rfdiffusion.inference.model_runners.sampler import Sampler


class SelfConditioning(Sampler):
    """
    Model Runner for self conditioning
    pX0[t+1] is provided as a template input to the model at time t
    """

    def sample_step(self, *, t: int, x_t: Tensor, seq_init: Tensor, final_step: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Generate the next pose that the model should be supplied at timestep t-1.
        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The sequence to the next step (== seq_init)
            plddt: (L, 1) Predicted lDDT of x0.
        """

        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = (
            self._preprocess(seq_init, x_t, t)
        )
        B, N, L = xyz_t.shape[:3]

        ##################################
        ######## Str Self Cond ###########
        ##################################
        if (t < self.diffuser.T) and (t != self.diffuser_conf.partial_T):
            zeros = torch.zeros(B, 1, L, 24, 3).float().to(xyz_t.device)
            xyz_t = torch.cat(
                (self.prev_pred.unsqueeze(1), zeros), dim=-2
            )  # [B,T,L,27,3]
            t2d_44 = xyz_to_t2d(xyz_t)  # [B,T,L,L,44]
        else:
            xyz_t = torch.zeros_like(xyz_t)
            t2d_44 = torch.zeros_like(t2d[..., :44])
        # No effect if t2d is only dim 44
        t2d[..., :44] = t2d_44

        if self.symmetry is not None:
            idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

        ####################
        ### Forward Pass ###
        ####################
        with torch.no_grad():
            msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(
                msa_masked,
                msa_full,
                seq_in,
                xt_in,
                idx_pdb,
                t1d=t1d,
                t2d=t2d,
                xyz_t=xyz_t,
                alpha_t=alpha_t,
                msa_prev=None,
                pair_prev=None,
                state_prev=None,
                t=torch.tensor(t),
                return_infer=True,
                motif_mask=self.diffusion_mask.squeeze().to(self.device),
                cyclic_reses=self.cyclic_reses,
            )

            if self.symmetry is not None and self.inf_conf.symmetric_self_cond:
                px0 = self.symmetrise_prev_pred(px0=px0, seq_in=seq_in, alpha=alpha)[
                    :, :, :3
                ]

        self.prev_pred = torch.clone(px0)

        # prediction of X0
        _, px0 = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0 = px0.squeeze()[:, :14]

        ###########################
        ### Generate Next Input ###
        ###########################

        seq_t_1 = torch.clone(seq_init)
        if t > final_step:
            x_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                align_motif=self.inf_conf.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
            )
            self._log.info(
                f"Timestep {t}, input to next step: { seq2chars(torch.argmax(seq_t_1, dim=-1).tolist())}"
            )
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            px0 = px0.to(x_t.device)

        ######################
        ### Apply symmetry ###
        ######################

        if self.symmetry is not None:
            x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)

        return px0, x_t_1, seq_t_1, plddt

    def symmetrise_prev_pred(self, px0, seq_in, alpha):
        """
        Method for symmetrising px0 output for self-conditioning
        """
        _, px0_aa = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0_sym, _ = self.symmetry.apply_symmetry(
            px0_aa.to("cpu").squeeze()[:, :14],
            torch.argmax(seq_in, dim=-1).squeeze().to("cpu"),
        )
        px0_sym = px0_sym[None].to(self.device)
        return px0_sym


class ScaffoldedSampler(SelfConditioning):
    """
    Model Runner for Scaffold-Constrained diffusion
    """

    def __init__(self, conf: DictConfig):
        """
        Initialize scaffolded sampler.
        Two basic approaches here:
            i) Given a block adjacency/secondary structure input, generate a fold (in the presence or absence of a target)
                - This allows easy generation of binders or specific folds
                - Allows simple expansion of an input, to sample different lengths
            ii) Providing a contig input and corresponding block adjacency/secondary structure input
                - This allows mixed motif scaffolding and fold-conditioning.
                - Adjacency/secondary structure inputs must correspond exactly in length to the contig string
        """
        super().__init__(conf)
        # initialize BlockAdjacency sampling class
        if conf.scaffoldguided.scaffold_dir is None:
            assert any(
                x is not None
                for x in (
                    conf.contigmap.inpaint_str_helix,
                    conf.contigmap.inpaint_str_strand,
                    conf.contigmap.inpaint_str_loop,
                )
            )
            if conf.contigmap.inpaint_str_loop is not None:
                assert (
                    conf.scaffoldguided.mask_loops == False
                ), "You shouldn't be masking loops if you're specifying loop secondary structure"
        else:
            # initialize BlockAdjacency sampling class
            assert all(
                x is None
                for x in (
                    conf.contigmap.inpaint_str_helix,
                    conf.contigmap.inpaint_str_strand,
                    conf.contigmap.inpaint_str_loop,
                )
            ), "can't provide scaffold_dir if you're also specifying per-residue ss"
            self.blockadjacency = BlockAdjacency(conf, conf.inference.num_designs)

        #################################################
        ### Initialize target, if doing binder design ###
        #################################################

        if conf.scaffoldguided.target_pdb:
            self.target = Target(conf.scaffoldguided, conf.ppi.hotspot_res)
            self.target_pdb = self.target.get_target()
            if conf.scaffoldguided.target_ss is not None:
                self.target_ss = torch.load(conf.scaffoldguided.target_ss).long()
                self.target_ss = torch.nn.functional.one_hot(
                    self.target_ss, num_classes=4
                )
                if self._conf.scaffoldguided.contig_crop is not None:
                    self.target_ss = self.target_ss[self.target_pdb["crop_mask"]]
            if conf.scaffoldguided.target_adj is not None:
                self.target_adj = torch.load(conf.scaffoldguided.target_adj).long()
                self.target_adj = torch.nn.functional.one_hot(
                    self.target_adj, num_classes=3
                )
                if self._conf.scaffoldguided.contig_crop is not None:
                    self.target_adj = self.target_adj[self.target_pdb["crop_mask"]]
                    self.target_adj = self.target_adj[:, self.target_pdb["crop_mask"]]
        else:
            self.target = None
            self.target_pdb = None

    def sample_init(self):
        """
        Wrapper method for taking secondary structure + adj, and outputting xt, seq_t
        """

        ##########################
        ### Process Fold Input ###
        ##########################
        if hasattr(self, "blockadjacency"):
            self.L, self.ss, self.adj = self.blockadjacency.get_scaffold()
            self.adj = nn.one_hot(self.adj.long(), num_classes=3)
        else:
            self.L = 100  # shim. Get's overwritten

        ##############################
        ### Auto-contig generation ###
        ##############################

        if self.contig_conf.contigs is None:
            # process target
            xT = torch.full((self.L, 27, 3), np.nan)
            xT = get_init_xyz(xT[None, None]).squeeze()
            seq_T = torch.full((self.L,), 21)
            self.diffusion_mask = torch.full((self.L,), False)
            atom_mask = torch.full((self.L, 27), False)
            self.binderlen = self.L

            if self.target:
                target_L = np.shape(self.target_pdb["xyz"])[0]
                # xyz
                target_xyz = torch.full((target_L, 27, 3), np.nan)
                target_xyz[:, :14, :] = torch.from_numpy(self.target_pdb["xyz"])
                xT = torch.cat((xT, target_xyz), dim=0)
                # seq
                seq_T = torch.cat(
                    (seq_T, torch.from_numpy(self.target_pdb["seq"])), dim=0
                )
                # diffusion mask
                self.diffusion_mask = torch.cat(
                    (self.diffusion_mask, torch.full((target_L,), True)), dim=0
                )
                # atom mask
                mask_27 = torch.full((target_L, 27), False)
                mask_27[:, :14] = torch.from_numpy(self.target_pdb["mask"])
                atom_mask = torch.cat((atom_mask, mask_27), dim=0)
                self.L += target_L
                # generate contigmap object
                contig = []
                for idx, i in enumerate(self.target_pdb["pdb_idx"][:-1]):
                    if idx == 0:
                        start = i[1]
                    if (
                        i[1] + 1 != self.target_pdb["pdb_idx"][idx + 1][1]
                        or i[0] != self.target_pdb["pdb_idx"][idx + 1][0]
                    ):
                        contig.append(f"{i[0]}{start}-{i[1]}/0 ")
                        start = self.target_pdb["pdb_idx"][idx + 1][1]
                contig.append(
                    f"{self.target_pdb['pdb_idx'][-1][0]}{start}-{self.target_pdb['pdb_idx'][-1][1]}/0 "
                )
                contig.append(f"{self.binderlen}-{self.binderlen}")
                contig = ["".join(contig)]
            else:
                contig = [f"{self.binderlen}-{self.binderlen}"]
            self.contig_map = ContigMap(self.target_pdb, contig)
            self.mappings = self.contig_map.get_mappings()
            self.mask_seq = self.diffusion_mask
            self.mask_str = self.diffusion_mask
            L_mapped = len(self.contig_map.ref)

        ############################
        ### Specific Contig mode ###
        ############################

        else:
            # get contigmap from command line
            assert (
                self.target is None
            ), "Giving a target is the wrong way of handling this is you're doing contigs and secondary structure"

            # process target and reinitialise potential_manager. This is here because the 'target' is always set up to be the second chain in out inputs.
            self.target_feats = process_target(self.inf_conf.input_pdb)
            self.contig_map = self.construct_contig(self.target_feats)
            self.mappings = self.contig_map.get_mappings()
            self.mask_seq = torch.from_numpy(self.contig_map.inpaint_seq)[None, :]
            self.mask_str = torch.from_numpy(self.contig_map.inpaint_str)[None, :]
            self.binderlen = len(self.contig_map.inpaint)
            self.L = len(self.contig_map.inpaint_seq)
            target_feats = self.target_feats
            contig_map = self.contig_map

            xyz_27 = target_feats["xyz_27"]
            mask_27 = target_feats["mask_27"]
            seq_orig = target_feats["seq"]
            L_mapped = len(self.contig_map.ref)
            seq_T = torch.full((L_mapped,), 21)
            seq_T[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]
            seq_T[~self.mask_seq.squeeze()] = 21

            diffusion_mask = self.mask_str
            self.diffusion_mask = diffusion_mask

            xT = torch.full((1, 1, L_mapped, 27, 3), np.nan)
            xT[:, :, contig_map.hal_idx0, ...] = xyz_27[contig_map.ref_idx0, ...]
            xT = get_init_xyz(xT).squeeze()
            atom_mask = torch.full((L_mapped, 27), False)
            atom_mask[contig_map.hal_idx0] = mask_27[contig_map.ref_idx0]

            if hasattr(self.contig_map, "ss_spec"):
                self.adj = torch.full((L_mapped, L_mapped), 2)  # masked
                self.adj = nn.one_hot(self.adj.long(), num_classes=3)
                self.ss = ss_from_contig(self.contig_map.ss_spec)
            assert L_mapped == self.adj.shape[0]

        ####################
        ### Get hotspots ###
        ####################
        self.hotspot_0idx = get_idx0_hotspots(
            self.mappings, self.ppi_conf, self.binderlen
        )

        #########################
        ### Set up potentials ###
        #########################

        self.potential_manager = PotentialManager(
            self.potential_conf,
            self.ppi_conf,
            self.diffuser_conf,
            self.inf_conf,
            self.hotspot_0idx,
            self.binderlen,
        )

        self.chain_idx = ["A" if i < self.binderlen else "B" for i in range(self.L)]

        ########################
        ### Handle Partial T ###
        ########################

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        t_list = np.arange(1, self.t_step_input + 1)
        seq_T = torch.nn.functional.one_hot(seq_T, num_classes=22).float()

        fa_stack, xyz_true = self.diffuser.diffuse_pose(
            xT,
            torch.clone(seq_T),
            atom_mask.squeeze(),
            diffusion_mask=self.diffusion_mask.squeeze(),
            t_list=t_list,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
        )

        #######################
        ### Set up Denoiser ###
        #######################

        self.denoiser = self.construct_denoiser(self.L, visible=self.mask_seq.squeeze())

        xT = torch.clone(fa_stack[-1].squeeze()[:, :14, :])

        ################################
        ### Add to Cyclic_reses init ###
        ################################
        self._init_cyclic_reses(self.mask_str, self.contig_map)

        return xT, seq_T

    def _preprocess(
        self, 
        seq, 
        xyz_t: Tensor, 
        t: LongTensor
        ) -> tuple[Tensor,Tensor,Tensor,Tensor,LongTensor,Tensor,Tensor,Tensor,Tensor]:
        msa_masked, msa_full, seq, xyz_prev, idx_pdb, t1d, t2d, xyz_t, alpha_t = (
            super()._preprocess(seq, xyz_t, t, repack=False)
        )

        ###################################
        ### Add Adj/Secondary Structure ###
        ###################################

        assert (
            self.preprocess_conf.d_t1d == 28
        ), "The checkpoint you're using hasn't been trained with sec-struc/block adjacency features"
        assert (
            self.preprocess_conf.d_t2d == 47
        ), "The checkpoint you're using hasn't been trained with sec-struc/block adjacency features"

        #####################
        ### Handle Target ###
        #####################

        if self.target:
            blank_ss = torch.nn.functional.one_hot(
                torch.full((self.L - self.binderlen,), 3), num_classes=4
            )
            full_ss = torch.cat((self.ss, blank_ss), dim=0)
            if self._conf.scaffoldguided.target_ss is not None:
                full_ss[self.binderlen :] = self.target_ss
        else:
            full_ss = self.ss
        t1d = torch.cat((t1d, full_ss[None, None].to(self.device)), dim=-1)

        t1d = t1d.float()

        ###########
        ### t2d ###
        ###########

        if self.d_t2d == 47:
            if self.target:
                full_adj = torch.zeros((self.L, self.L, 3))
                full_adj[:, :, -1] = 1.0  # set to mask
                full_adj[: self.binderlen, : self.binderlen] = self.adj
                if self._conf.scaffoldguided.target_adj is not None:
                    full_adj[self.binderlen :, self.binderlen :] = self.target_adj
            else:
                full_adj = self.adj
            t2d = torch.cat((t2d, full_adj[None, None].to(self.device)), dim=-1)

        ###########
        ### idx ###
        ###########

        if self.target:
            idx_pdb[:, self.binderlen :] += 200

        return msa_masked, msa_full, seq, xyz_prev, idx_pdb, t1d, t2d, xyz_t, alpha_t
