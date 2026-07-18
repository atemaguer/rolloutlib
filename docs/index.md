# rolloutlib

Rolloutlib is a collection of interoperable Python libraries for agentic
reinforcement-learning (RL) post-training.

It defines standard APIs for the interactive part of agentic RL: environments
define tasks and state transitions; multimodal spaces describe observations and
actions; policies sample language-model actions; rollout runners record
trajectories; and graders produce reward and evaluation signals.

These contracts keep task, trajectory, and reward semantics portable across
model providers, rollout workers, RL trainers, and evaluation harnesses.
Existing Gymnasium environments remain normal Gymnasium environments, and
language-facing wrappers and policies compose around them.

## The interaction model

An environment presents an observation and accepts an action. A policy samples
the action. A rollout records the interaction. A grader validates its declared
input and turns an action, trajectory, tool trace, or other application value
into a structured score.

```text
task example
    → fresh environment
    → policy actions
    → trajectory
    → score
    → training backend
```

These pieces stay independent. The same environment can be used for evaluation
or training; the same grader can grade a single response or a collected
trajectory; the same trajectory can be adapted to different optimizers.

That separation is the core interoperability promise: integrations translate
at stable boundaries instead of reimplementing the task, agent loop, or grading
logic for every backend.

## Concepts

- [Environments](concepts/environments.md) define tasks and state transitions.
- [Spaces](concepts/spaces.md) describe valid observations, actions, and grader
  inputs.
- [Policies and rollouts](concepts/rollouts.md) sample and record interactions.
- [Graders](concepts/graders.md) define portable, composable grading signals.
- [Evaluations](concepts/evaluations.md) run user-provided examples and
  aggregate task behavior.

The grader documentation includes focused guides to
[rubrics](graders/rubrics.md), [programmatic rewards](graders/reward-graders.md),
[composition](graders/composite-graders.md), [structured
scores](graders/scores-and-aggregation.md), and [representative
examples](graders/examples.md).

Start with [Getting started](getting-started.md) for installation and a small
end-to-end example. Use the [API reference](api.md) when implementing against a
specific interface.
