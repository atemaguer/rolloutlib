Core concepts
=============

Environments
------------

Synchronous environments implement the Gymnasium ``reset``/``step`` contract.
``AsyncEnv`` provides the same value-level contract for asynchronous tools and
resources. Grading that determines reward belongs inside ``step``; the complete
structured score is retained in the environment ``info`` mapping.

Spaces
------

Rolloutlib accepts every Gymnasium space and supplies common spaces for token
IDs and sequences, text, messages, chat, and tool calls. Spaces remain
role-neutral: the same space can describe an action or an observation.

Policies and rollouts
---------------------

``Policy`` is the synchronous callable contract and ``AsyncPolicy`` is its
async-compatible counterpart. A policy may return a raw action or
``PolicyOutput`` with generated tokens, log probabilities, and stop metadata.
``rollout`` and ``arollout`` record environment interactions as ``Trajectory``
objects without owning model sampling.

Datasets and graders
--------------------

``Dataset`` and ``RLDataset`` describe pre-rollout task items and fresh
environment factories. Rubrics describe what should be evaluated; graders
produce scalar or structured ``Score`` values that environments can use as
rewards.

Evaluation
----------

Benchmarks own task items and environment factories. Evaluation callbacks run
fresh environments and aggregate the scores those environments produce. This
keeps benchmark evaluation on the same environment semantics used by RL
rollouts.
