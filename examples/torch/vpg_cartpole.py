#!/usr/bin/env python3
"""
This is an example to train a task with VPG algorithm.

Here it runs CartPole-v1 environment with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 16
"""
import torch
import torch.nn.functional as F

from garage.experiment import LocalRunner, run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.envs import TfEnv
from garage.torch.algos import VPG
from garage.torch.modules import MLPModule
from garage.torch.policies import DeterministicPolicy


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task."""
    with LocalRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(env_name='CartPole-v1')

        policy_module = MLPModule(
            input_dim=env.spec.observation_space.flat_dim,
            output_dim=env.spec.action_space.flat_dim,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=F.relu,
            output_nonlinearity=torch.tanh)
        policy = DeterministicPolicy(env.spec, policy_module)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = VPG(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            optimizer_args=dict(tf_optimizer_args=dict(learning_rate=0.01, )))

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=10000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
