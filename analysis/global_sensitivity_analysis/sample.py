from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    """Class containing parameter values from a single sample for global sensitivity analysis"""
    max_speed: float
    vision: float
    soc_strength: float
    obs_strength: float
