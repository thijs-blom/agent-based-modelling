from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    max_speed: float
    vision: float
    soc_strength: float
    obs_strength: float
