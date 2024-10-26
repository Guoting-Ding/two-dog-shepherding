from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class SheepLowdimRunner(BaseLowdimRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseLowdimPolicy):
        return dict()
