# Environments

An environment defines the task that a policy interacts with. Rolloutlib uses
Gymnasium's environment contract so existing Gymnasium tools and expectations
continue to apply.

## Synchronous environments

A synchronous environment is a `gymnasium.Env`. Its two central operations are:

- `reset()` starts an episode and returns the first observation plus an
  information dictionary.
- `step(action)` advances the task and returns the next observation, scalar
  reward, `terminated`, `truncated`, and information.

`terminated` means the task reached an outcome. `truncated` means an external
limit ended the episode. Keeping those meanings distinct matters to training
algorithms.

## Asynchronous environments

`AsyncEnv` preserves the same observations, actions, rewards, termination
flags, and information while making `reset`, `step`, and `close` awaitable.
It is useful for tool execution, remote services, browsers, and other
asynchronous resources.

`as_async` lifts a Gymnasium environment into the async convention without
blocking the event loop. `as_sync` exposes an async environment through the
synchronous Gymnasium API. Calls on a single adapted instance are serialized;
parallelism belongs across independent environment instances.

## Single-turn tasks

`SingleTurnEnv` and `AsyncSingleTurnEnv` cover tasks where one action completes
the episode. Subclasses define:

- the initial observation;
- how the action is evaluated;
- the terminal observation.

The evaluator may return a scalar or a structured `Score`. Structured scores
become the scalar reward while remaining available under `info["score"]`.

## Adding grading to an environment

Grading that determines reward belongs inside `step`. If an existing
environment does not grade its own terminal state, `GradingWrapper` or
`AsyncGradingWrapper` can compose it with a grader.

```python
from rolloutlib import GradingWrapper

environment = GradingWrapper(
    ExistingEnv(item),
    grader=grader,
    make_input=lambda env, action: (item, env.state, action),
)
```

By default, grading occurs on terminated or truncated steps, replaces the
environment reward with `Score.value`, and retains the full score in `info`.
Custom predicates and reward combiners can change those policies. When using a
rubric grader, its rubric is already bound to the grader rather than passed
through the environment wrapper.

See [Composite graders](../graders/composite-graders.md#grading-inside-environments)
for synchronous and asynchronous wrapper examples, terminal-step policies, and
reward combination.
