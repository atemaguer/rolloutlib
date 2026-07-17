# rolloutlib

Rolloutlib defines small, composable interfaces for agentic and language-model
reinforcement-learning post-training.

The library standardizes the boundaries between a task, a policy, a recorded
interaction, and a grading signal. It deliberately leaves model sampling,
optimization algorithms, distributed execution, and backend-specific training
records to the systems that own them.

## The interaction model

An environment presents an observation and accepts an action. A policy samples
the action. A rollout records the interaction. A grader validates its declared
input and turns an action, trajectory, tool trace, or other application value
into a structured score.

```text
dataset item
    → fresh environment
    → policy actions
    → trajectory
    → score
    → training backend
```

These pieces stay independent. The same environment can be used for evaluation
or training; the same grader can grade a single response or a collected
trajectory; the same trajectory can be adapted to different optimizers.

## Concepts

- [Environments](concepts/environments.md) define tasks and state transitions.
- [Spaces](concepts/spaces.md) describe valid observations, actions, and grader
  inputs.
- [Policies and rollouts](concepts/rollouts.md) sample and record interactions.
- [Graders and rubrics](concepts/graders.md) define portable grading signals.
- [Datasets and evaluation](concepts/datasets-and-evaluation.md) organize work
  and benchmark task behavior.

Start with [Getting started](getting-started.md) for installation and a small
end-to-end example. Use the [API reference](api.md) when implementing against a
specific interface.
