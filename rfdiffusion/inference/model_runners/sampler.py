from typing import Any

import string
import logging
from pathlib import Path

import torch
from torch import Tensor, BoolTensor, LongTensor
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import torch.nn.functional as nn

from rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
from rfdiffusion.kinematics import get_init_xyz, xyz_to_t2d
from rfdiffusion.diffusion import Diffuser
from rfdiffusion.chemical import seq2chars
from rfdiffusion.util_module import ComputeAllAtomCoords
from rfdiffusion.contigs import ContigMap
from rfdiffusion.potentials.manager import PotentialManager
from rfdiffusion.util import get_torsions_initialized
from rfdiffusion.model_input_logger import pickle_function_call

from rfdiffusion.inference.symmetry import SymGen
from rfdiffusion.inference.utils import Denoise,process_target, get_idx0_hotspots

PACKAGE_DIR = Path(__file__).parents[3]
EXAMPLES_DIR = PACKAGE_DIR / "examples"
MODELS_DIR = PACKAGE_DIR / "models"
SCHEDULES_DIR = PACKAGE_DIR / "schedules"

class Sampler:

    def __init__(self, conf: DictConfig):
        """
        Initialize sampler.
        Args:
            conf: Configuration.
        """
        self.initialized = False
        self.initialize(conf)

    def initialize(self, conf: DictConfig) -> None:
        """
        Initialize sampler.
        Args:
            conf: Configuration

        - Selects appropriate model from input
        - Assembles Config from model checkpoint and command line overrides

        """
        self._log = logging.getLogger(__name__)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        needs_model_reload = (
            not self.initialized
            or conf.inference.ckpt_override_path
            != self._conf.inference.ckpt_override_path
        )

        # Assign config to Sampler
        self._conf = conf

        ################################
        ### Select Appropriate Model ###
        ################################

        if conf.inference.model_directory_path is not None:
            model_directory = Path(conf.inference.model_directory_path)
        else:
            model_directory = MODELS_DIR

        print(f"Reading models from {model_directory}")

        # Initialize inference only helper objects to Sampler
        if conf.inference.ckpt_override_path is not None:
            self.ckpt_path = conf.inference.ckpt_override_path
            print(
                "WARNING: You're overriding the checkpoint path from the defaults. Check that the model you're providing can run with the inputs you're providing."
            )
        else:
            if (
                conf.contigmap.inpaint_seq is not None
                or conf.contigmap.provide_seq is not None
                or conf.contigmap.inpaint_str
            ):
                # use model trained for inpaint_seq
                if conf.contigmap.provide_seq is not None:
                    # this is only used for partial diffusion
                    assert (
                        conf.diffuser.partial_T is not None
                    ), "The provide_seq input is specifically for partial diffusion"
                if conf.scaffoldguided.scaffoldguided:
                    self.ckpt_path = model_directory / "InpaintSeq_Fold_ckpt.pt"
                else:
                    self.ckpt_path = model_directory / "InpaintSeq_ckpt.pt"
            elif (
                conf.ppi.hotspot_res is not None
                and not conf.scaffoldguided.scaffoldguided
            ):
                # use complex trained model
                self.ckpt_path = model_directory / "Complex_base_ckpt.pt"
            elif conf.scaffoldguided.scaffoldguided:
                # use complex and secondary structure-guided model
                self.ckpt_path = model_directory / "Complex_Fold_base_ckpt.pt"
            else:
                # use default model
                self.ckpt_path = model_directory / "Base_ckpt.pt"
        # for saving in trb file:
        assert (
            self._conf.inference.trb_save_ckpt_path is None
        ), "trb_save_ckpt_path is not the place to specify an input model. Specify in inference.ckpt_override_path"
        self._conf["inference"]["trb_save_ckpt_path"] = self.ckpt_path

        #######################
        ### Assemble Config ###
        #######################

        if needs_model_reload:
            # Load checkpoint, so that we can assemble the config
            self.load_checkpoint()
            self.assemble_config_from_chk()
            # Now actually load the model weights into RF
            self.model = self.load_model()
        else:
            self.assemble_config_from_chk()

        # self.initialize_sampler(conf)
        self.initialized = True

        # Initialize helper objects
        self.inf_conf = self._conf.inference
        self.contig_conf = self._conf.contigmap
        self.denoiser_conf = self._conf.denoiser
        self.ppi_conf = self._conf.ppi
        self.potential_conf = self._conf.potentials
        self.diffuser_conf = self._conf.diffuser
        self.preprocess_conf = self._conf.preprocess

        if conf.inference.schedule_directory_path is not None:
            schedule_directory = Path(conf.inference.schedule_directory_path)
        else:
            schedule_directory = SCHEDULES_DIR

        # Check for cache schedule
        if not schedule_directory.exists():
            schedule_directory.mkdir()
        self.diffuser = Diffuser(**self._conf.diffuser, cache_dir=str(schedule_directory))

        ###########################
        ### Initialise Symmetry ###
        ###########################

        if self.inf_conf.symmetry is not None:
            self.symmetry = SymGen(
                self.inf_conf.symmetry,
                self.inf_conf.recenter,
                self.inf_conf.radius,
                self.inf_conf.model_only_neighbors,
            )
        else:
            self.symmetry = None

        self.allatom = ComputeAllAtomCoords().to(self.device)

        if self.inf_conf.input_pdb is None:
            # set default pdb
            self.inf_conf.input_pdb = str(EXAMPLES_DIR / "input_pdbs" / "1qys.pdb")
        self.target_feats = process_target(
            self.inf_conf.input_pdb, parse_hetatom=True, center=False
        )
        self.chain_idx = None
        self.idx_pdb = None

        ##############################
        ### Handle Partial Noising ###
        ##############################

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)

    @property
    def T(self) -> int:
        """
        Return the maximum number of timesteps
        that this design protocol will perform.

        Output:
            T (int): The maximum number of timesteps to perform
        """
        return self.diffuser_conf.T

    def load_checkpoint(self) -> None:
        """Loads RF checkpoint, from which config can be generated."""
        self._log.info(f"Reading checkpoint from {self.ckpt_path}")
        print("This is inf_conf.ckpt_path")
        print(self.ckpt_path)
        self.ckpt = torch.load(self.ckpt_path, map_location=self.device)

    def assemble_config_from_chk(self) -> None:
        """
        Function for loading model config from checkpoint directly.

        Takes:
            - config file

        Actions:
            - Replaces all -model and -diffuser items
            - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint

        This throws an error if there is a flag in the checkpoint 'config_dict' that isn't in the inference config.
        This should ensure that whenever a feature is added in the training setup, it is accounted for in the inference script.

        """
        # get overrides to re-apply after building the config from the checkpoint
        overrides = []
        if HydraConfig.initialized():
            overrides = HydraConfig.get().overrides.task
        print("Assembling -model, -diffuser and -preprocess configs from checkpoint")

        for cat in ["model", "diffuser", "preprocess"]:
            for key in self._conf[cat]:
                try:
                    print(
                        f"USING MODEL CONFIG: self._conf[{cat}][{key}] = {self.ckpt['config_dict'][cat][key]}"
                    )
                    self._conf[cat][key] = self.ckpt["config_dict"][cat][key]
                except:
                    pass

        # add overrides back in again
        for override in overrides:
            if override.split(".")[0] in ["model", "diffuser", "preprocess"]:
                print(
                    f'WARNING: You are changing {override.split("=")[0]} from the value this model was trained with. Are you sure you know what you are doing?'
                )
                mytype = type(
                    self._conf[override.split(".")[0]][
                        override.split(".")[1].split("=")[0]
                    ]
                )
                self._conf[override.split(".")[0]][
                    override.split(".")[1].split("=")[0]
                ] = mytype(override.split("=")[1])

    def load_model(self) -> RoseTTAFoldModule:
        """Create RosettaFold model from preloaded checkpoint."""

        # Read input dimensions from checkpoint.
        self.d_t1d = self._conf.preprocess.d_t1d
        self.d_t2d = self._conf.preprocess.d_t2d
        model = RoseTTAFoldModule(
            **self._conf.model,
            d_t1d=self.d_t1d,
            d_t2d=self.d_t2d,
            T=self._conf.diffuser.T,
        ).to(self.device)
        if self._conf.logging.inputs:
            pickle_dir = pickle_function_call(model, "forward", "inference")
            print(f"pickle_dir: {pickle_dir}")
        model = model.eval()
        self._log.info(f"Loading checkpoint.")
        model.load_state_dict(self.ckpt["model_state_dict"], strict=True)
        return model

    def construct_contig(self, target_feats) -> ContigMap:
        """
        Construct contig class describing the protein to be generated
        """
        self._log.info(f"Using contig: {self.contig_conf.contigs}")
        return ContigMap(target_feats, **self.contig_conf)

    def construct_denoiser(self, L: int, visible: bool):
        """Make length-specific denoiser."""
        denoise_kwargs: dict[str, Any] = OmegaConf.to_container(self.diffuser_conf) 
        denoise_kwargs |= {
                "L": L,
                "diffuser": self.diffuser,
                "potential_manager": self.potential_manager,
            }
        return Denoise(**denoise_kwargs)

    def sample_init(self, return_forward_trajectory: bool = False) -> tuple[Tensor, Tensor]:
        """
        Initial features to start the sampling process.

        Modify signature and function body for different initialization
        based on the config.

        Returns:
            xt: Starting positions with a portion of them randomly sampled.
            seq_t: Starting sequence with a portion of them set to unknown.
        """

        #######################
        ### Parse input pdb ###
        #######################

        self.target_feats = process_target(
            self.inf_conf.input_pdb, parse_hetatom=True, center=False
        )

        ################################
        ### Generate specific contig ###
        ################################

        # Generate a specific contig from the range of possibilities specified at input

        self.contig_map = self.construct_contig(self.target_feats)
        self.mappings = self.contig_map.get_mappings()
        self.mask_seq = torch.from_numpy(self.contig_map.inpaint_seq)[None, :]
        self.mask_str = torch.from_numpy(self.contig_map.inpaint_str)[None, :]
        self.binderlen = len(self.contig_map.inpaint)
        #######################################
        ### Resolve cyclic peptide indicies ###
        #######################################
        self._init_cyclic_reses(self.mask_str, self.contig_map)

        ####################
        ### Get Hotspots ###
        ####################

        self.hotspot_0idx = get_idx0_hotspots(
            self.mappings, self.ppi_conf, self.binderlen
        )

        #####################################
        ### Initialise Potentials Manager ###
        #####################################

        self.potential_manager = PotentialManager(
            self.potential_conf,
            self.ppi_conf,
            self.diffuser_conf,
            self.inf_conf,
            self.hotspot_0idx,
            self.binderlen,
        )

        ###################################
        ### Initialize other attributes ###
        ###################################

        xyz_27 = self.target_feats["xyz_27"]
        mask_27 = self.target_feats["mask_27"]
        seq_orig = self.target_feats["seq"]
        L_mapped = len(self.contig_map.ref)
        contig_map = self.contig_map

        self.diffusion_mask = self.mask_str
        length_bound = self.contig_map.sampled_mask_length_bound.copy()

        first_res = 0
        self.chain_idx = []
        self.idx_pdb = []
        all_chains = {contig_ref[0] for contig_ref in self.contig_map.ref}
        available_chains = sorted(list(set(string.ascii_letters) - all_chains))

        # Iterate over each chain
        for last_res in length_bound:
            chain_ids = {
                contig_ref[0] for contig_ref in self.contig_map.ref[first_res:last_res]
            }
            # If we are designing this chain, it will have a '-' in the contig map
            # Renumber this chain from 1
            if "_" in chain_ids:
                self.idx_pdb += [idx + 1 for idx in range(last_res - first_res)]
                chain_ids = chain_ids - {"_"}
                # If there are no fixed residues that have a chain id, pick the first available letter
                if not chain_ids:
                    if not available_chains:
                        raise ValueError(
                            f"No available chains!  You are trying to design a new chain, and you have "
                            f"already used all upper- and lower-case chain ids (up to 52 chains): "
                            f"{','.join(all_chains)}."
                        )
                    chain_id = available_chains[0]
                    available_chains.remove(chain_id)
                # Otherwise, use the chain of the fixed (motif) residues
                else:
                    assert (
                        len(chain_ids) == 1
                    ), f"Error: Multiple chain IDs in chain: {chain_ids}"
                    chain_id = list(chain_ids)[0]
                self.chain_idx += [chain_id] * (last_res - first_res)
            # If this is a fixed chain, maintain the chain and residue numbering
            else:
                self.idx_pdb += [
                    contig_ref[1]
                    for contig_ref in self.contig_map.ref[first_res:last_res]
                ]
                assert (
                    len(chain_ids) == 1
                ), f"Error: Multiple chain IDs in chain: {chain_ids}"
                self.chain_idx += [list(chain_ids)[0]] * (last_res - first_res)
            first_res = last_res

        ####################################
        ### Generate initial coordinates ###
        ####################################

        if self.diffuser_conf.partial_T:
            assert (
                xyz_27.shape[0] == L_mapped
            ), f"there must be a coordinate in the input PDB for \
                    each residue implied by the contig string for partial diffusion.  length of \
                    input PDB != length of contig string: {xyz_27.shape[0]} != {L_mapped}"
            assert (
                contig_map.hal_idx0 == contig_map.ref_idx0
            ), f"for partial diffusion there can \
                    be no offset between the index of a residue in the input and the index of the \
                    residue in the output, {contig_map.hal_idx0} != {contig_map.ref_idx0}"
            # Partially diffusing from a known structure
            xyz_mapped = xyz_27
            atom_mask_mapped = mask_27
        else:
            # Fully diffusing from points initialised at the origin
            # adjust size of input xt according to residue map
            xyz_mapped = torch.full((1, 1, L_mapped, 27, 3), np.nan)
            xyz_mapped[:, :, contig_map.hal_idx0, ...] = xyz_27[
                contig_map.ref_idx0, ...
            ]
            xyz_motif_prealign = xyz_mapped.clone()
            motif_prealign_com = xyz_motif_prealign[0, 0, :, 1].mean(dim=0)
            self.motif_com = xyz_27[contig_map.ref_idx0, 1].mean(dim=0)
            xyz_mapped = get_init_xyz(xyz_mapped).squeeze()
            # adjust the size of the input atom map
            atom_mask_mapped = torch.full((L_mapped, 27), False)
            atom_mask_mapped[contig_map.hal_idx0] = mask_27[contig_map.ref_idx0]

        # Diffuse the contig-mapped coordinates
        if self.diffuser_conf.partial_T:
            assert (
                self.diffuser_conf.partial_T <= self.diffuser_conf.T
            ), "Partial_T must be less than T"
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        t_list = np.arange(1, self.t_step_input + 1)

        #################################
        ### Generate initial sequence ###
        #################################

        seq_t = torch.full((1, L_mapped), 21).squeeze()  # 21 is the mask token
        seq_t[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]

        # Unmask sequence if desired
        if self._conf.contigmap.provide_seq is not None:
            seq_t[self.mask_seq.squeeze()] = seq_orig[self.mask_seq.squeeze()]

        seq_t[~self.mask_seq.squeeze()] = 21
        seq_t = torch.nn.functional.one_hot(seq_t, num_classes=22).float()  # [L,22]
        seq_orig = torch.nn.functional.one_hot(
            seq_orig, num_classes=22
        ).float()  # [L,22]

        fa_stack, xyz_true = self.diffuser.diffuse_pose(
            xyz_mapped,
            torch.clone(seq_t),
            atom_mask_mapped.squeeze(),
            diffusion_mask=self.diffusion_mask.squeeze(),
            t_list=t_list,
        )
        xT = fa_stack[-1].squeeze()[:, :14, :]
        xt = torch.clone(xT)

        self.denoiser = self.construct_denoiser(
            len(self.contig_map.ref), visible=self.mask_seq.squeeze()
        )

        ######################
        ### Apply Symmetry ###
        ######################

        if self.symmetry is not None:
            xt, seq_t = self.symmetry.apply_symmetry(xt, seq_t)
        self._log.info(f"Sequence init: {seq2chars(torch.argmax(seq_t, dim=-1))}")

        self.msa_prev = None
        self.pair_prev = None
        self.state_prev = None

        #########################################
        ### Parse ligand for ligand potential ###
        #########################################

        if self.potential_conf.guiding_potentials is not None:
            if any(
                list(
                    filter(
                        lambda x: "substrate_contacts" in x,
                        self.potential_conf.guiding_potentials,
                    )
                )
            ):
                assert (
                    len(self.target_feats["xyz_het"]) > 0
                ), "If you're using the Substrate Contact potential, \
                        you need to make sure there's a ligand in the input_pdb file!"
                het_names = np.array(
                    [i["name"].strip() for i in self.target_feats["info_het"]]
                )
                xyz_het = self.target_feats["xyz_het"][
                    het_names == self._conf.potentials.substrate
                ]
                xyz_het = torch.from_numpy(xyz_het)
                assert (
                    xyz_het.shape[0] > 0
                ), f"expected >0 heteroatoms from ligand with name {self._conf.potentials.substrate}"
                xyz_motif_prealign = xyz_motif_prealign[0, 0][
                    self.diffusion_mask.squeeze()
                ]
                for pot in self.potential_manager.potentials_to_apply:
                    pot.motif_substrate_atoms = xyz_het
                    pot.diffusion_mask = self.diffusion_mask.squeeze()
                    pot.xyz_motif = xyz_motif_prealign
                    pot.diffuser = self.diffuser
        return xt, seq_t

    def _init_cyclic_reses(self, mask_str: BoolTensor, contig_map: ContigMap):
        """
        Centralized logic for initializing self.cyclic_reses.

        mask_str: tensor-like mask (can be 1D or have a leading batch dim), where True indicates motif/fixed.
        contig_map: object with attribute 'hal' (iterable of per-residue chain ids).
        """
        mask = mask_str.squeeze()
        # ensure on correct device and dtype
        mask = mask.to(self.device)

        if self._conf.inference.cyclic:
            if self._conf.inference.cyc_chains is None:
                self.cyclic_reses = ~mask
            else:
                assert isinstance(
                    self._conf.inference.cyc_chains, str
                ), "cyc_chains arg must be string"
                cyc_chains = [i.upper() for i in self._conf.inference.cyc_chains]
                hal_idx = contig_map.hal
                is_cyclized = torch.zeros_like(mask).bool().to(self.device).squeeze()
                # build boolean mask per residue based on chain id in hal_idx
                ch_mask_list = [idx[0] for idx in hal_idx]
                for ch in cyc_chains:
                    ch_mask = torch.tensor(
                        [c == ch for c in ch_mask_list],
                        dtype=torch.bool,
                        device=self.device,
                    )
                    is_cyclized[ch_mask] = True
                self.cyclic_reses = is_cyclized
        else:
            self.cyclic_reses = torch.zeros_like(mask).bool().to(self.device).squeeze()

    def _preprocess(
        self, 
        seq: Tensor, 
        xyz_t: Tensor, 
        t: LongTensor, 
        repack: bool = False,
        ) -> tuple[Tensor,Tensor,Tensor,Tensor,LongTensor,Tensor,Tensor,Tensor,Tensor]:
        """
        Function to prepare inputs to diffusion model

            seq (L,22) one-hot sequence

            msa_masked (1,1,L,48)

            msa_full (1,1,L,25)

            xyz_t (L,14,3) template crds (diffused)

            t1d (1,L,28) this is the t1d before tacking on the chi angles:
                - seq + unknown/mask (21)
                - global timestep (1-t/T if not motif else 1) (1)

                MODEL SPECIFIC:
                - contacting residues: for ppi. Target residues in contact with binder (1)
                - empty feature (legacy) (1)
                - ss (H, E, L, MASK) (4)

            t2d (1, L, L, 45)
                - last plane is block adjacency
        """

        L = seq.shape[0]

        ##################
        ### msa_masked ###
        ##################
        msa_masked = torch.zeros((1, 1, L, 48))
        msa_masked[:, :, :, :22] = seq[None, None]
        msa_masked[:, :, :, 22:44] = seq[None, None]
        msa_masked[:, :, 0, 46] = 1.0
        msa_masked[:, :, -1, 47] = 1.0

        ################
        ### msa_full ###
        ################
        msa_full = torch.zeros((1, 1, L, 25))
        msa_full[:, :, :, :22] = seq[None, None]
        msa_full[:, :, 0, 23] = 1.0
        msa_full[:, :, -1, 24] = 1.0

        ###########
        ### t1d ###
        ###########

        # Here we need to go from one hot with 22 classes to one hot with 21 classes (last plane is missing token)
        t1d = torch.zeros((1, 1, L, 21))

        seqt1d = torch.clone(seq)
        for idx in range(L):
            if seqt1d[idx, 21] == 1:
                seqt1d[idx, 20] = 1
                seqt1d[idx, 21] = 0

        t1d[:, :, :, :21] = seqt1d[None, None, :, :21]

        # Set timestep feature to 1 where diffusion mask is True, else 1-t/T
        timefeature = torch.zeros((L)).float()
        timefeature[self.mask_str.squeeze()] = 1
        timefeature[~self.mask_str.squeeze()] = 1 - t / self.T
        timefeature = timefeature[None, None, ..., None]

        t1d = torch.cat((t1d, timefeature), dim=-1).float()

        #############
        ### xyz_t ###
        #############
        if self.preprocess_conf.sidechain_input:
            xyz_t[torch.where(seq == 21, True, False), 3:, :] = float("nan")
        else:
            xyz_t[~self.mask_str.squeeze(), 3:, :] = float("nan")

        xyz_t = xyz_t[None, None]
        xyz_t = torch.cat((xyz_t, torch.full((1, 1, L, 13, 3), float("nan"))), dim=3)

        ###########
        ### t2d ###
        ###########
        t2d = xyz_to_t2d(xyz_t)

        ###########
        ### idx ###
        ###########
        idx = torch.tensor(self.contig_map.rf)[None]

        ###############
        ### alpha_t ###
        ###############
        seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
        alpha, _, alpha_mask, _ = get_torsions_initialized(xyz_t.reshape(-1, L, 27, 3), seq_tmp)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[..., 0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1, -1, L, 10, 2)
        alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)

        # put tensors on device
        msa_masked = msa_masked.to(self.device)
        msa_full = msa_full.to(self.device)
        seq = seq.to(self.device)
        xyz_t = xyz_t.to(self.device)
        idx = idx.to(self.device)
        t1d = t1d.to(self.device)
        t2d = t2d.to(self.device)
        alpha_t = alpha_t.to(self.device)

        ######################
        ### added_features ###
        ######################
        if self.preprocess_conf.d_t1d >= 24:  # add hotspot residues
            hotspot_tens = torch.zeros(L).float()
            if self.ppi_conf.hotspot_res is None:
                print(
                    "WARNING: you're using a model trained on complexes and hotspot residues, without specifying hotspots.\
                         If you're doing monomer diffusion this is fine"
                )
                hotspot_idx = []
            else:
                hotspots = [(i[0], int(i[1:])) for i in self.ppi_conf.hotspot_res]
                hotspot_idx = []
                for i, res in enumerate(self.contig_map.con_ref_pdb_idx):
                    if res in hotspots:
                        hotspot_idx.append(self.contig_map.hal_idx0[i])
                hotspot_tens[hotspot_idx] = 1.0

            # Add blank (legacy) feature and hotspot tensor
            t1d = torch.cat(
                (
                    t1d,
                    torch.zeros_like(t1d[..., :1]),
                    hotspot_tens[None, None, ..., None].to(self.device),
                ),
                dim=-1,
            )

        return (
            msa_masked,
            msa_full,
            seq[None],
            torch.squeeze(xyz_t, dim=0),
            idx,
            t1d,
            t2d,
            xyz_t,
            alpha_t,
        )

    def sample_step(self, *, t: int, x_t: Tensor, seq_init: Tensor, final_step: int) -> tuple[Tensor,Tensor,Tensor,Tensor]:
        """Generate the next pose that the model should be supplied at timestep t-1.

        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.

        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L,22) The updated sequence of the next step.
            plddt: (L, 1) Predicted lDDT of x0.
        """
        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = (
            self._preprocess(seq_init, x_t, t)
        )

        if self.symmetry is not None:
            idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

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
            )

        # prediction of X0
        _, px0 = self.allatom.forward(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0 = px0.squeeze()[:, :14]

        #####################
        ### Get next pose ###
        #####################

        if t > final_step:
            seq_t_1 = nn.one_hot(seq_init, num_classes=22).to(self.device)
            x_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                align_motif=self.inf_conf.align_motif,
            )
        else:
            x_t_1 = px0.to(x_t.device, copy=True)
            seq_t_1 = torch.clone(seq_init)
            px0 = px0.to(x_t.device)

        if self.symmetry is not None:
            x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)

        return px0, x_t_1, seq_t_1, plddt

