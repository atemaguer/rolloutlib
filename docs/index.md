# rolloutlib

Gymnasium-style environments and rollout primitives for agentic RL
post-training.

Rolloutlib provides small, composable contracts for environments, spaces,
policies, rollouts, datasets, graders, and evaluation. It does not own model
sampling or a training loop; those remain user- or backend-specific.

## What belongs in rolloutlib

- Gymnasium-compatible synchronous environments and async counterparts.
- Token, text, message, chat, and tool-call spaces.
- Sync and async policy contracts and trajectory collection.
- RL datasets, grouped rollouts, structured scores, rubrics, and graders.
- Model-backend-neutral benchmark evaluation.

Start with [Getting started](getting-started.md), then explore the
[core concepts](concepts.md) and [API reference](api.md).
