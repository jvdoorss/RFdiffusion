import os

from omegaconf import DictConfig

from .sampler import Sampler
from .scaffolded_sampler import ScaffoldedSampler, SelfConditioning
from .runner import SamplerWrapper

def sampler_selector(conf: DictConfig) -> Sampler:
    if conf.scaffoldguided.scaffoldguided:
        return ScaffoldedSampler(conf)
    match conf.inference.model_runner:
        case "default":
            return Sampler(conf)
        case "SelfConditioning":
            return SelfConditioning(conf)
        case "ScaffoldedSampler":
            return ScaffoldedSampler(conf)
        case _ as sampler:
            raise ValueError(f"Unrecognized sampler {sampler}")