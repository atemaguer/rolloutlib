"""Opt-in live test for a multimodal OpenAI agent playing 20 chess plies.

Run with ``RUN_OPENAI_CHESS_INTEGRATION=1`` after installing the
``openai-chess`` extra and configuring ``OPENAI_API_KEY``. The test makes 20
paid OpenAI Responses API requests.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, cast

import gymnasium as gym
import pytest


if os.getenv("RUN_OPENAI_CHESS_INTEGRATION") != "1":
    pytest.skip(
        "set RUN_OPENAI_CHESS_INTEGRATION=1 to run the paid OpenAI chess test",
        allow_module_level=True,
    )

chess = importlib.import_module("chess")
importlib.import_module("gym_bullet_chess")
openai = importlib.import_module("openai")

from rolloutlib import rollout, wrappers  # noqa: E402
from rolloutlib.policies.openai import OpenAIResponsesPolicy  # noqa: E402


def test_openai_agent_plays_twenty_legal_moves_in_gymnasium_chess() -> None:
    model = os.getenv("OPENAI_CHESS_MODEL", "gpt-5.6-luna")
    max_steps = 20
    native = gym.make(
        "BulletChess-v0",
        self_play=True,
        capture_visual=True,
    )
    chess_env = cast(Any, native.unwrapped)

    def uci_to_native(uci: str) -> int:
        move = chess.Move.from_uci(uci)
        board = chess_env.board
        if move not in board.legal_moves:
            raise ValueError(f"illegal move: {uci}")
        return int(move.from_square * 64 + move.to_square)

    uci_env = gym.wrappers.TransformAction(
        native,
        uci_to_native,
        action_space=gym.spaces.Text(min_length=4, max_length=5),
    )

    def state(_: dict[str, Any]) -> dict[str, Any]:
        board = chess_env.board
        return {
            "turn": "white" if board.turn == chess.WHITE else "black",
            "fen": board.fen(),
            "legal_moves": [move.uci() for move in board.legal_moves],
        }

    env = wrappers.wrap_language_env(
        uci_env,
        state=state,
        image=lambda observation: observation["board_img"],
        image_alt="Current chess board",
        tool_name="play_move",
        argument_name="uci",
        tool_description="Play one legal chess move in UCI notation.",
        available_actions=lambda: [
            move.uci() for move in chess_env.board.legal_moves
        ],
    )
    policy = OpenAIResponsesPolicy.from_env(
        env,
        client=openai.OpenAI(),
        model=model,
        reasoning={"effort": "none"},
        instructions=(
            "Play one legal chess move for the side to move. "
            "Inspect the board image and structured state, then call "
            "play_move exactly once."
        ),
        image_detail="low",
        max_output_tokens=512,
        store=False,
    )

    try:
        trajectory = rollout(
            env,
            policy,
            seed=7,
            max_steps=max_steps,
            metadata={"model": model, "environment": "BulletChess-v0"},
        )
    finally:
        env.close()

    assert len(trajectory.steps) == max_steps
    assert len(chess_env.board.move_stack) == max_steps
    played_moves = [move.uci() for move in chess_env.board.move_stack]
    for step, board_move in zip(
        trajectory.steps,
        chess_env.board.move_stack,
        strict=True,
    ):
        move = cast(str, step.action["arguments"]["uci"])
        assert step.action["name"] == "play_move"
        assert board_move.uci() == move
        assert step.reward != -10.0
        assert step.next_observation in env.observation_space
        assert step.policy_info["response_id"].startswith("resp_")
        assert "gpt-5.6-luna" in step.policy_info["model"]
        assert step.policy_info["reasoning_effort"] == "none"
    assert trajectory.truncated

    print(
        "OpenAI chess validation: "
        f"model={model}, reasoning=none, plies={len(played_moves)}, "
        f"moves={' '.join(played_moves)}, final_fen={chess_env.board.fen()}"
    )

    initial_content = trajectory.initial_observation[0]["content"]
    assert isinstance(initial_content, list)
    image_part = next(part for part in initial_content if part["type"] == "image")
    assert image_part["url"].startswith("data:image/png;base64,")
