"""Vanilla Policy Gradient."""
import numpy as np
import torch
from torch.nn import functional as F  # noqa

from garage.misc import special
from garage.np.algos import RLAlgorithm
from garage.tf.misc import tensor_utils


class VPG(RLAlgorithm):
    """Vanilla Policy Gradient.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_path_length (int): Maximum length of a single rollout.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in garage.tf.optimizers.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
    """

    def __init__(
            self,
            env_spec,  # TODO NO USE
            policy,
            baseline,
            scope=None,  # TODO NO USE
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            center_adv=True,
            positive_adv=False,
            fixed_horizon=False,  # TODO NO USE
            max_kl_step=0.01,
            optimizer=None,
            optimizer_args=None,  # TODO ??
            policy_ent_coeff=0.0,
            stop_entropy_gradient=False,
            entropy_method='no_entropy',
    ):
        self._env_spec = env_spec
        self._policy = policy
        self._baseline = baseline
        self._scope = scope
        self._max_path_length = max_path_length
        self._discount = discount
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._fixed_horizon = fixed_horizon
        self._max_kl_step = max_kl_step
        self._optimizer = optimizer
        self._optimizer_args = optimizer_args
        self._policy_ent_coeff = policy_ent_coeff
        self._stop_entropy_gradient = stop_entropy_gradient
        self._entropy_method = entropy_method
        self._eps = 1e-8

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        #     Entropy regularization, max entropy rl

        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)

    def _check_entropy_configuration(self, entropy_method, center_adv,
                                     stop_entropy_gradient, policy_ent_coeff):
        if entropy_method not in ('max', 'regularized', 'no_entropy'):
            raise ValueError('Invalid entropy_method')

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
        if entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')

    def train_once(self, itr, paths):
        """Perform one step of policy optimization."""
        self._optimize_policy(paths)
        self._optimize_baseline(paths)

    def _optimize_policy(self, paths):
        loss = [self._get_policy_loss(path) for path in paths]

        self._optimizer.zero_grad()
        torch.cat(loss).mean().backward()
        self._optimizer.step()

    def _get_policy_loss(self, path):
        # TODO nograd?

        # TODO self._maximum_entropy -> difference?
        policy_entropy = self._policy.distribution.entropy()
        rewards = torch.Tensor(path['rewards'])

        if self._maximum_entropy:
            rewards = [
                reward + self._policy_ent_coeff * policy_entropy
                for reward in rewards
            ]

        baselines = self._get_baselines(path)
        advantages = self._compute_advantages(baselines, rewards)

        if self._center_adv:
            mean, var = advantages.mean(), advantages.var()
            advantages = F.batch_norm(advantages, mean, var, eps=self._eps)

        if self._positive_adv:
            advantages -= advantages.min()

        # TODO kl -> pol_mean_kl ??

        log_likelihood = self._policy.distribution.log_prob(path['actions'])
        advantages = log_likelihood.sum() * advantages.sum()

        if self._entropy_regularzied:
            advantages += self._policy_ent_coeff * policy_entropy

        return advantages

    def _compute_advantages(self, baselines, rewards):
        path_baselines = F.pad(baselines, (0, 1), value=0)
        deltas = (rewards + self._discount * path_baselines[1:] -
                  path_baselines[:-1])
        return special.discount_cumsum(deltas,
                                       self._discount * self._gae_lambda)

    def _get_baselines(self, path):
        return torch.Tensor((self.baseline.predict_n(path) if hasattr(
            self.baseline, 'predict_n') else self.baseline.predict(path)))

    def _pad_with_zero(self, tensor, max_length):
        padding_size = max(max_length - tensor.shape[1], 0)
        return F.pad(tensor, padding_size, value=0)

    def _optimize_baseline(self, paths):
        max_path_length = self.max_path_length

        #  TODO Baseline needs advantage?
        # for idx, path in enumerate(paths):
        #     path['baselines'] = self._get_baselines(path)
        #
        #     path_baselines = np.append(path['baselines'], 0)
        #     path['deltas'] = (path['rewards'] +
        #                       self.discount * path_baselines[1:] -
        #                       path_baselines[:-1])
        #
        #     path['advantages'] = special.discount_cumsum(
        #         path['deltas'], self.discount * self.gae_lambda)
        #
        #     path['returns'] = special.discount_cumsum(
        #         path['rewards'], self.discount)
        #

        valids = [np.ones_like(path['returns']) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        rewards = self._collect_element_with_padding(paths, 'rewards',
                                                     max_path_length),
        rewards = [
            rew[val.astype(np.bool)] for rew, val in zip(rewards, valids)
        ]

        returns = self._collect_element_with_padding(paths, 'returns',
                                                     max_path_length),
        returns = [
            ret[val.astype(np.bool)] for ret, val in zip(returns, valids)
        ]

        agent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(path['agent_infos'], max_path_length)
            for path in paths
        ])

        env_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(path['env_infos'], max_path_length)
            for path in paths
        ])

        samples_data = dict(
            observations=self._collect_element_with_padding(
                paths, 'observations', max_path_length),
            actions=self._collect_element_with_padding(paths, 'actions',
                                                       max_path_length),
            baselines=self._collect_element_with_padding(
                paths, 'baselines', max_path_length),
            rewards=rewards,
            returns=returns,
            valids=valids,
            agent_infos=agent_infos,
            env_infos=env_infos,
            average_return=np.mean(sum(path['rewards']) for path in paths),
            paths=paths,
        )

        if hasattr(self.baseline, 'fit_with_samples'):
            self.baseline.fit_with_samples(paths, samples_data)
        else:
            self.baseline.fit(paths)

    def train(self, runner, batch_size):
        """Obtain samplers and start actual training for each epoch."""
        pass
