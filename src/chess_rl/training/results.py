from dataclasses import dataclass

import chess
import torch


@dataclass
class StepResult:
    """
    Result of one complete agent-environment interaction.
    """

    state: torch.Tensor
    action: int
    move: chess.Move
    reward: float
    next_state: torch.Tensor
    done: bool
    info: dict


@dataclass
class EpisodeResult:
    """
    Result of one complete episode.

    An episode normally represents one chess game, although it can
    also stop early when the configured step limit is reached.
    """

    steps: int
    total_reward: float
    done: bool
    truncated: bool
    final_info: dict


@dataclass
class VsRandomEpisodeResult:
    """
    Result of one episode where the DQN plays White
    and RandomAgent plays Black.
    """

    agent_steps: int
    total_plies: int
    total_reward: float
    done: bool
    truncated: bool
    final_info: dict
    training_losses: list[float]
    final_epsilon: float
    replay_size: int


@dataclass
class TrainingSummary:
    """
    Aggregate metrics from a multi-episode training run.
    """

    episodes: int
    average_reward: float
    average_loss: float | None
    final_epsilon: float
    replay_size: int


@dataclass
class EvaluationSummary:
    """
    Aggregate results from evaluation games against RandomAgent.
    """

    episodes: int
    wins: int
    draws: int
    losses: int
    truncated: int