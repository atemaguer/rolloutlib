# Policies and rollouts

A policy chooses an action from an observation. A rollout records what happened
when a policy interacted with an environment.

Rolloutlib does not own model sampling. `Policy` and `AsyncPolicy` are callable
contracts that application code can implement using a hosted API, a local
model, or a training backend.

## Policy outputs

A policy may return a raw action or `PolicyOutput`. `PolicyOutput` associates
the semantic action with model-side sampling information:

```python
from rolloutlib import PolicyOutput


def policy(observation):
    result = model.generate(observation)
    return PolicyOutput(
        action=result.text,
        tokens=result.tokens,
        logprobs=result.logprobs,
        stop_reason=result.stop_reason,
    )
```

Environment information and policy information remain separate. The
environment owns `Step.info`; the sampler owns `Step.policy_info` and the
structured policy sampling fields.

## Trajectories

`rollout` and `arollout` collect one episode into a `Trajectory`. Each `Step`
records the observation, action, reward, next observation, termination flags,
environment information, and policy information.

The rollout functions do not close the environment supplied by the caller.
They may apply a collection-level `max_steps` truncation.

## Groups

`rollout_group` and `arollout_group` collect several independent trajectories
for one source item. The environment factory is called once per trajectory, and
the group collectors own and close those fresh environments.

`TrajectoryGroup.scores` retains structured scores. `TrajectoryGroup.rewards`
provides their scalar values for optimizer adapters. Async groups bound
concurrency across independent environment instances.
