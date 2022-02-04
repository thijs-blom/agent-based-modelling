from analysis.global_sensitivity_analysis.sample import Sample
from model.oneexit import OneExit


class OneExitSample(OneExit):
    """Model for evacuation through a single exit, with a sample parameter for global sensitivity analysis"""
    def __init__(self, sample: Sample):
        """
        Args:
            sample: A sample from the parameter space used for global sensitivity analysis
        """
        super().__init__(
            max_speed=sample.max_speed,
            vision=sample.vision,
            soc_strength=sample.soc_strength,
            obs_strength=sample.obs_strength,
        )
