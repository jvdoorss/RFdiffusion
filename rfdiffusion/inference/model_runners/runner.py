'''
Module defining sample iterating functionality
'''

from dataclasses import dataclass

import torch
from torch import Tensor
from omegaconf import DictConfig

from rfdiffusion.contigs import ContigMap
from rfdiffusion.inference.model_runners.sampler import Sampler

@dataclass 
class SamplerState:
    t: int
    x : torch.Tensor
    seq: torch.Tensor

    px0: torch.Tensor | None = None
    plddt: torch.Tensor | None = None

class BaseSampleIterator:
    def __init__(self, sampler: Sampler, step: int = 1):
        self.sampler: Sampler = sampler
        self.step: int = step 
        self.final_step: int = sampler.inf_conf.final_step

        t = sampler.t_step_input
        x_init, seq_init = self.sampler.sample_init()
        self.current = SamplerState(t,x_init,seq_init)

    def __next__(self) -> SamplerState:
        if self.current.t <= self.final_step:
            raise StopIteration
        px0, x, seq, plddt = self.sampler.sample_step(t=self.current.t,
                                                      x_t = self.current.x,
                                                      seq_init=self.current.seq,
                                                      final_step = self.final_step)
        self.current = SamplerState(self.current.t-self.step,x,seq, px0, plddt)
        return self.current


class SamplerWrapper(Sampler):
    '''
    Baseclass for a SampleWrapper, implementing all attributes required for inference.
    '''

    def __init__(self, base_sampler: Sampler):
        self.sampler = base_sampler
        self.seq_init = None

    @property
    def inf_conf(self) -> DictConfig:
        return self.sampler.inf_conf

    @property
    def t_step_input(self) -> int:
        return self.sampler.t_step_input

    @property
    def binderlen(self) -> int:
        return self.sampler.binderlen

    @property
    def chain_idx(self) -> list[int] | None:
        return self.sampler.chain_idx

    @property
    def idx_pdb(self) -> list | None:
        return self.sampler.idx_pdb

    @property
    def _conf(self) -> DictConfig:
        return self.sampler._conf

    @property
    def contig_map(self) -> ContigMap:
        return self.sampler.contig_map

    def __iter__(self) -> BaseSampleIterator:
        iterator = BaseSampleIterator(self.sampler)
        self.seq_init = iterator.current.seq # store 
        return iterator

    def run(self):
        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []

        # Loop over number of reverse diffusion time steps.
        for state in self:
            px0_xyz_stack.append(state.px0)
            denoised_xyz_stack.append(state.x)
            seq_stack.append(state.seq)
            plddt_stack.append(state.plddt[0])  # remove singleton leading dimension

        # Flip order for better visualization in pymol
        denoised_xyz_stack = torch.flip(torch.stack(denoised_xyz_stack),(0,))
        px0_xyz_stack = torch.flip(torch.stack(px0_xyz_stack),(0,))

        # For logging -- don't flip
        plddt_stack = torch.stack(plddt_stack)

        return px0_xyz_stack, denoised_xyz_stack, seq_stack, plddt_stack