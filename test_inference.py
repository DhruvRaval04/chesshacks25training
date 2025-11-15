#!/usr/bin/env python3
"""Quick smoke-test script to run inference with a trained chess policy."""

from __future__ import annotations

import random
from pathlib import Path

import chess
import torch

from rl_chess import ChessEnv
from rl_chess.models import PolicyValueNet
from rl_chess.move_encoding import MAX_MOVES, index_to_move, move_to_index


MODEL_ID = "draval/chesshacks"
CACHE_DIR = Path("./.model_cache").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = min(5, MAX_MOVES)
MAX_RANDOM_PLIES = 20

# Load the trained policy once on import.


def _load_policy() -> PolicyValueNet:
    model: PolicyValueNet
    if MODEL_ID:
        try:
            model = PolicyValueNet.from_pretrained(
                MODEL_ID,
                cache_dir=str(CACHE_DIR),
            ).to(DEVICE)
            model.eval()
            print(f"[policy] Loaded from Hugging Face repo '{MODEL_ID}'.")
            return model
        except OSError as err:
            print(
                "[policy] Failed to load from Hugging Face "
                f"('{MODEL_ID}'): {err}"
            )


policy = _load_policy()


def prepare_random_observation(
    env: ChessEnv,
    max_random_plies: int = MAX_RANDOM_PLIES,
) -> tuple[torch.Tensor, torch.Tensor]:
    env.reset()
    random_plies = random.randint(0, max_random_plies)
    for _ in range(random_plies):
        if env.board.is_game_over():
            break
        legal_moves = list(env.board.legal_moves)
        if not legal_moves:
            break
        env.board.push(random.choice(legal_moves))
    obs = env._get_obs()
    mask = env._legal_moves_mask()
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(DEVICE)
    return obs_tensor, mask_tensor


def pretty_print_results(
    board: chess.Board,
    action_idx: int,
    log_prob: float,
    value: float,
    probs: torch.Tensor,
    top_k: int,
) -> None:
    print("\n=== Board State ===")
    print(board)
    print(f"FEN: {board.fen()}")

    move = index_to_move(action_idx)
    print("\n=== Model Output ===")
    print(
        "Chosen move index: "
        f"{action_idx} -> {move.uci()} "
        f"(verify {move_to_index(move)})"
    )
    print(f"Log probability: {log_prob:.4f}")
    print(f"Value head prediction: {value:.4f}")

    k = min(top_k, probs.numel())
    top_probs, top_indices = torch.topk(probs, k)
    print("\nTop moves:")
    for rank, (prob, idx) in enumerate(
        zip(top_probs.tolist(), top_indices.tolist()),
        start=1,
    ):
        print(
            f"  {rank}. {index_to_move(idx).uci():5s} "
            f"prob={prob:.4f} index={idx}"
        )


def run_demo() -> None:
    env = ChessEnv(agent_color=chess.WHITE)
    obs_tensor, mask_tensor = prepare_random_observation(env)

    with torch.no_grad():
        logits, value = policy(obs_tensor, mask_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action_tensor = torch.argmax(dist.logits, dim=-1)
        log_prob = dist.log_prob(action_tensor).item()
        action_idx = action_tensor.item()
        probs = torch.softmax(dist.logits, dim=-1).squeeze(0).cpu()
        value_scalar = value.squeeze(0).item()

    pretty_print_results(
        board=env.board,
        action_idx=action_idx,
        log_prob=log_prob,
        value=value_scalar,
        probs=probs,
        top_k=TOP_K,
    )


if __name__ == "__main__":
    run_demo()
