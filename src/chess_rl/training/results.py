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
    Result of one episode 
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
    claimable_threefold: bool = False
    claimable_fifty_moves: bool = False
    agent_color: chess.Color = chess.WHITE
    final_material_balance: int = 0
    final_total_material: int = 78


@dataclass
class TrainingSummary:
    """
    Aggregate metrics from a multi-episode training run.
    """

    episodes: int
    wins: int
    draws: int
    losses: int
    truncated: int
    average_plies: float
    average_reward: float
    average_loss: float | None
    final_epsilon: float
    replay_size: int
    truncated_claimable_threefold: int = 0
    truncated_claimable_fifty_moves: int = 0
    truncated_without_claimable_draw: int = 0
    truncated_average_total_material: float | None = None
    truncated_average_material_balance: float | None = None
    truncated_average_absolute_material_balance: float | None = None


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
    truncated_claimable_threefold: int = 0
    truncated_claimable_fifty_moves: int = 0
    truncated_without_claimable_draw: int = 0
    truncated_average_total_material: float | None = None
    truncated_average_material_balance: float | None = None
    truncated_average_absolute_material_balance: float | None = None