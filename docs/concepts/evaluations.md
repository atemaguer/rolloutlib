# Evaluations

Evaluation measures agent behavior using the same environments, trajectories,
and structured scores used during post-training. Rolloutlib standardizes that
evaluation boundary without defining how example collections are represented.

Examples and their storage remain application-owned. A `Benchmark` only binds a
name and a user-provided sequence of examples to a factory that creates a fresh
environment for each example:

```python
from rolloutlib.evals import Benchmark

benchmark = Benchmark(
    name="customer-support",
    items=examples,
    make_env=lambda example: SupportEnv(example),
    item_id=lambda example: example.id,
)
```

The evaluation callback owns policy interaction and returns an `Evaluation`
containing a structured `Score`. The runner handles environment lifecycles,
failures, limits, and aggregation:

```python
from rolloutlib import rollout
from rolloutlib.evals import Evaluation, run_benchmark
from rolloutlib.graders import Score


def evaluate(environment):
    trajectory = rollout(environment, policy)
    return Evaluation(
        score=Score.from_info(
            trajectory.steps[-1].info,
            default=Score(trajectory.total_reward),
        ),
        truncated=trajectory.truncated,
    )


result = run_benchmark(benchmark, evaluate)
```

Rolloutlib does not prescribe how examples are loaded, stored, streamed,
shuffled, or batched. Those concerns belong to the application or training
system. The standard interface begins when an example becomes an environment
and continues through the resulting trajectory and score.
