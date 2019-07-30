"""GaussianMLPPolicy."""

from garage.torch.policies import Policy


class GaussianMLPPolicy(Policy):
    """
    GaussianMLPPolicy.

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        module : GaussianMLPModule to make prediction based on a gaussian
        distribution.
    :return:

    """

    def __init__(self, env_spec, module):
        self._module = module
        self.distribution = None

    def forward(self, inputs):
        """Forward method."""
        return self._module(inputs)

    def get_actions(self, observations):
        """Get actions given observations."""
        self.distribution = self.forward(observations)
        return self.distribution.rsample().detach().numpy()
