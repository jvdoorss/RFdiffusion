#!/usr/bin/env python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import re
import os, time, pickle
import logging
import random
import glob

import torch
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np

from rfdiffusion.util import writepdb_multi, writepdb
from rfdiffusion.inference.model_runners import sampler_selector, SamplerWrapper, Sampler


torch.set_float32_matmul_precision('high')


def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def log_device_info(log: logging.Logger):
    # Check for available GPU and print result of check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

def next_fileindex(prefix: str, extension: str = '.pdb') -> int:
    '''Get first unused index for the given file prefix and extension'''
    existing = glob.glob(prefix + "*" + extension)
    indices = [-1]
    for e in existing:
        if m := re.match(r".*_(\d+)\.$", e):
            m = m.groups()[0]
            indices.append(int(m))
    return max(indices) + 1

@hydra.main(version_base=None, config_path="../config/inference", config_name="base")
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)
    log_device_info(log)

    # Initialize sampler and target/contig.
    if conf.inference.deterministic:
        make_deterministic()
    sampler = sampler_selector(conf)

    # Loop over number of designs to sample.
    design_startnum = sampler.inf_conf.design_startnum
    if sampler.inf_conf.design_startnum == -1:
        design_startnum = next_fileindex(sampler.inf_conf.output_prefix, ".pdb")

    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        create_design(i_des, sampler, conf, log)


def create_design(i_des: int, sampler: Sampler, conf: HydraConfig, log: logging.Logger):

    if conf.inference.deterministic:
        make_deterministic(i_des)

    start_time = time.time()
    out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
    log.info(f"Making design {out_prefix}")
    if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
        log.info(
            f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists."
        )
        return

    sample_runner = SamplerWrapper(sampler)

    px0_xyz_stack, denoised_xyz_stack, _, plddt_stack = sample_runner.run()
    seq_final = sample_runner.seq_init
    final_seq, bfacts = postprocess_sequence(seq_final)

    # Save outputs
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    out = f"{out_prefix}.pdb"

    # Now don't output sidechains
    writepdb(
        out,
        denoised_xyz_stack[0, :, :4],
        final_seq,
        sampler.binderlen,
        chain_idx=sampler.chain_idx,
        bfacts=bfacts,
        idx_pdb=sampler.idx_pdb
    )

    run_metadata(sampler, plddt_stack, start_time, out_prefix)

    if sampler.inf_conf.write_trajectory:
        write_trajectory(out_prefix, denoised_xyz_stack, bfacts, final_seq, px0_xyz_stack, sampler.chain_idx)

    if conf.inference.empty_cache_per_design and torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info(f"Finished design in {(time.time()-start_time)/60:.2f} minutes")

def postprocess_sequence(sequence: torch.Tensor) -> tuple[torch.LongTensor, torch.Tensor]:
    '''
    Convert/Revert sequence from categorical to indexed (selecting the maximal)
    
    * replace residues outside the motif region by glycine
    * create b factors as:
        - 1.0 for the motif
        - 0.0 for diffused residues

    Parameters:
    -----------
    sequence:   torch.Tensor, shape (*,21)
                2D tensor with weights for each residue category

    Returns:
    --------
    tuple:
        - torch.LongTensor of shape (*,)
        - torch.Tensor containing 0/1 of shape (*,) 
    '''
    # Output glycines, except for motif region
    final_seq = torch.argmax(sequence, dim=-1)
    non_motif = final_seq == 21 
    final_seq[non_motif] = 7 # 7 is glycine

    bfacts = torch.ones_like(final_seq.squeeze())
    # make bfact=0 for diffused coordinates
    bfacts[non_motif] = 0

    return final_seq, bfacts

def run_metadata(sampler: Sampler, plddt_stack: torch.Tensor, start_time: float, out_prefix: str):
    # run metadata
    trb = dict(
        config=OmegaConf.to_container(sampler._conf, resolve=True),
        plddt=plddt_stack.cpu().numpy(),
        device=torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else "CPU",
        time=time.time() - start_time,
    )
    if hasattr(sampler, "contig_map"):
        for key, value in sampler.contig_map.get_mappings().items():
            trb[key] = value
    with open(f"{out_prefix}.trb", "wb") as f_out:
        pickle.dump(trb, f_out)

def write_trajectory(out_prefix: str, 
                    denoised_xyz_stack: torch.Tensor, 
                    bfacts: torch.Tensor, 
                    final_seq: torch.Tensor, 
                    px0_xyz_stack: torch.Tensor, 
                    chain_idx: list[int]):
    '''trajectory pdbs'''
    traj_prefix = (
        os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
    )
    os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

    out = f"{traj_prefix}_Xt-1_traj.pdb"
    writepdb_multi(
        out,
        denoised_xyz_stack,
        bfacts,
        final_seq.squeeze(),
        use_hydrogens=False,
        backbone_only=False,
        chain_ids=chain_idx,
    )

    out = f"{traj_prefix}_pX0_traj.pdb"
    writepdb_multi(
        out,
        px0_xyz_stack,
        bfacts,
        final_seq.squeeze(),
        use_hydrogens=False,
        backbone_only=False,
        chain_ids=chain_idx,
    )

if __name__ == "__main__":
    main()
