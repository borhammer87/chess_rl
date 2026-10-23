import random
from time import perf_counter
import numpy as np
import torch
import argparse
import chess
import chess.pgn
from chess_rl.agents.random_agent import RandomAgent
from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
from chess_rl.env.chess_env import ChessEnv
from chess_rl.training.train_dqn import (
    evaluate_against_random_both_colors,
    score_evaluation,
    summarize_training,
    train_against_random,
    save_greedy_evaluation_game,
    get_episode_outcome,
    )
from chess_rl.training.episodes import PER_ALPHA
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from chess_rl.utils.replay_buffer import ReplayBuffer, Transition
from chess_rl.utils.board_encoder import encode_board
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.action_selection import (
    evaluate_state_action_greedy_choice,
)

def set_random_seed(seed: int) -> None:
    """Seed Python and PyTorch random number generators."""
    random.seed(seed)
    torch.manual_seed(seed)

class TerminalTransitionTracker:
    """Track episode-ending transitions without modifying replay data."""

    def __init__(self, replay_buffer: ReplayBuffer) -> None:
        self.replay_buffer = replay_buffer
        self.labels = {}

    def record_episode(
        self,
        completed_episodes: int,
        total_episodes: int,
        result,
    ) -> None:
        outcome = get_episode_outcome(result)

        if outcome == "unfinished":
            raise ValueError(
                "Cannot label an unfinished training episode."
            )

        if not self.replay_buffer.buffer:
            raise ValueError(
                "Completed episode has no replay transitions."
            )

        transition = self.replay_buffer.buffer[-1]

        if not transition.done and not result.truncated:
            raise ValueError(
                "Last replay transition is neither terminal "
                "nor artificially truncated."
            )

        self.labels[id(transition)] = (
            transition,
            outcome,
        )

    def get_final_labels(self) -> dict[int, str]:
        """Return labels for terminal transitions still in replay."""
        labels = {}

        for transition in self.replay_buffer.buffer:
            if (
                not transition.done
                and id(transition) not in self.labels
            ):
                continue

            entry = self.labels.get(id(transition))

            if entry is None or entry[0] is not transition:
                raise ValueError(
                    "Unlabelled terminal transition in final replay."
                )

            labels[id(transition)] = entry[1]

        return labels


class PrioritizedSampleObserver:
    """Record actual PER selections without changing sampling."""

    def __init__(self, replay_buffer: ReplayBuffer) -> None:
        self.original_sample = replay_buffer.sample_prioritized
        self.sampled_transitions: list[Transition] = []

    def sample_prioritized(
        self,
        batch_size: int,
        alpha: float,
        beta: float,
    ) -> tuple[list[Transition], list[int], torch.Tensor]:
        result = self.original_sample(
            batch_size=batch_size,
            alpha=alpha,
            beta=beta,
        )

        transitions, _, _ = result
        self.sampled_transitions.extend(transitions)

        return result


def analyze_observed_per_samples(
    observer: PrioritizedSampleObserver,
    tracker: TerminalTransitionTracker,
) -> None:
    """Summarize actual PER selections during training."""
    counts = Counter()

    for transition in observer.sampled_transitions:
        entry = tracker.labels.get(id(transition))

        if entry is not None and entry[0] is transition:
            category = entry[1]
        elif transition.done:
            raise ValueError(
                "Sampled terminal transition has no label."
            )
        else:
            category = "non-terminal"

        counts[category] += 1

    total = len(observer.sampled_transitions)
    print(f"\nObserved PER selections - total: {total}")

    for category in (
        "non-terminal", "truncated", "win", "loss", "draw"
    ):
        count = counts[category]
        share = count / total if total else 0.0
        print(f"{category}: count={count} - share={share:.6f}")

def print_diagnostic_pgn(path: Path) -> None:
    """Include a complete diagnostic game in the console log."""
    print("\n--- BEGIN DIAGNOSTIC PGN ---")
    print(path.read_text(encoding="utf-8").strip())
    print("--- END DIAGNOSTIC PGN ---")

def analyze_greedy_pgn(
    path: Path,
    agent: StateActionDQNAgent,
    agent_color: chess.Color,
) -> None:
    """Compare recorded learner moves with reconstructed greedy choices."""
    with path.open(
        encoding="utf-8",
    ) as pgn_file:
        game = chess.pgn.read_game(
            pgn_file
        )

    if game is None:
        raise ValueError(
            "PGN does not contain a game."
        )

    board = game.board()
    learner_move_number = 0

    for move in game.mainline_moves():
        if board.turn == agent_color:
            learner_move_number += 1

            state = encode_board(
                board
            ).to(agent.device)
            legal_moves = list(
                board.legal_moves
            )

            (
                greedy_action,
                best_q,
                q_gap,
            ) = evaluate_state_action_greedy_choice(
                network=agent.policy_net,
                state=state,
                legal_moves=legal_moves,
            )

            recorded_action = encode_move(
                move
            )

            if greedy_action != recorded_action:
                raise RuntimeError(
                    "Reconstructed greedy action does not "
                    "match recorded PGN move "
                    f"at learner move {learner_move_number}: "
                    f"recorded={move.uci()}, "
                    f"greedy_action={greedy_action}."
                )

            legal_actions = [
                encode_move(legal_move)
                for legal_move in legal_moves
            ]

            with torch.no_grad():
                q_values = agent.policy_net.evaluate_action_ids(
                    state=state,
                    action_ids=legal_actions,
                )

            q_values = q_values.detach().cpu().float()

            min_q = q_values.min().item()
            mean_q = q_values.mean().item()
            std_q = q_values.std(
                unbiased=False
            ).item()
            q_range = best_q - min_q

            q_gap_text = (
                f"{q_gap:.6f}"
                if q_gap is not None
                else "N/A"
            )

            print(
                f"{learner_move_number:3d}. "
                f"{move.uci()} "
                f"- legal: {len(legal_moves)} "
                f"- best: {best_q:.6f} "
                f"- gap: {q_gap_text} "
                f"- mean: {mean_q:.6f} "
                f"- min: {min_q:.6f} "
                f"- std: {std_q:.6f} "
                f"- range: {q_range:.6f}"
            )

        board.push(
            move
        )

def analyze_replay_learning_signal(
    agent: StateActionDQNAgent,
    replay_buffer: ReplayBuffer,
    batch_size: int = 256,
) -> None:
    """Summarize Bellman targets and TD errors in the final replay."""
    transitions = list(
        replay_buffer.buffer
    )

    if not transitions:
        raise ValueError(
            "Cannot analyze an empty replay buffer."
        )

    all_rewards = []
    all_targets = []
    all_q_values = []
    all_abs_td_errors = []
    all_done = []

    with torch.no_grad():
        for start in range(
            0,
            len(transitions),
            batch_size,
        ):
            batch = transitions[
                start:start + batch_size
            ]

            states = torch.stack([
                transition.state
                for transition in batch
            ]).to(agent.device)

            actions = [
                transition.action
                for transition in batch
            ]

            rewards = torch.tensor(
                [
                    transition.reward
                    for transition in batch
                ],
                dtype=torch.float32,
                device=agent.device,
            )

            q_values = (
                agent.policy_net.evaluate_state_action_pairs(
                    states,
                    actions,
                )
            )

            next_q_values = torch.zeros(
                len(batch),
                dtype=torch.float32,
                device=agent.device,
            )

            non_terminal_indices = [
                index
                for index, transition in enumerate(batch)
                if not transition.done
            ]

            for index in non_terminal_indices:
                if not batch[index].next_legal_actions:
                    raise ValueError(
                        "Non-terminal transition must have "
                        "legal next actions."
                    )

            if non_terminal_indices:
                next_states = torch.stack([
                    batch[index].next_state
                    for index in non_terminal_indices
                ]).to(agent.device)

                next_legal_actions = [
                    batch[index].next_legal_actions
                    for index in non_terminal_indices
                ]

                next_q_values[
                    non_terminal_indices
                ] = (
                    agent.target_net.evaluate_legal_action_maxes(
                        next_states,
                        next_legal_actions,
                    )
                )

            targets = (
                rewards
                + agent.gamma * next_q_values
            )

            abs_td_errors = (
                targets - q_values
            ).abs()

            all_rewards.append(
                rewards.cpu()
            )
            all_targets.append(
                targets.cpu()
            )
            all_q_values.append(
                q_values.cpu()
            )
            all_abs_td_errors.append(
                abs_td_errors.cpu()
            )
            all_done.extend(
                transition.done
                for transition in batch
            )

    rewards = torch.cat(
        all_rewards
    )
    targets = torch.cat(
        all_targets
    )
    q_values = torch.cat(
        all_q_values
    )
    abs_td_errors = torch.cat(
        all_abs_td_errors
    )
    done_mask = torch.tensor(
        all_done,
        dtype=torch.bool,
    )

    def print_stats(
        label: str,
        values: torch.Tensor,
    ) -> None:
        print(
            f"{label}: "
            f"mean={values.mean().item():.6f} "
            f"- std={values.std(unbiased=False).item():.6f} "
            f"- min={values.min().item():.6f} "
            f"- max={values.max().item():.6f}"
        )

    print(
        f"Replay transitions: {len(transitions)} "
        f"- non-terminal: {(~done_mask).sum().item()} "
        f"- terminal: {done_mask.sum().item()}"
    )

    print_stats(
        "Reward",
        rewards,
    )
    print_stats(
        "Bellman target",
        targets,
    )
    print_stats(
        "Q(s,a)",
        q_values,
    )
    print_stats(
        "|TD error|",
        abs_td_errors,
    )

    if done_mask.any():
        print("\nTerminal transitions:")
        print_stats(
            "Reward",
            rewards[done_mask],
        )
        print_stats(
            "Bellman target",
            targets[done_mask],
        )
        print_stats(
            "Q(s,a)",
            q_values[done_mask],
        )
        print_stats(
            "|TD error|",
            abs_td_errors[done_mask],
        )

    non_terminal_mask = ~done_mask

    if non_terminal_mask.any():
        print("\nNon-terminal transitions:")
        print_stats(
            "Reward",
            rewards[non_terminal_mask],
        )
        print_stats(
            "Bellman target",
            targets[non_terminal_mask],
        )
        print_stats(
            "Q(s,a)",
            q_values[non_terminal_mask],
        )
        print_stats(
            "|TD error|",
            abs_td_errors[non_terminal_mask],
        )


def analyze_replay_priorities(
    replay_buffer: ReplayBuffer,
) -> None:
    """Summarize PER weighting of terminal replay transitions."""
    transitions = list(replay_buffer.buffer)
    priorities = np.asarray(
        list(replay_buffer.priorities),
        dtype=np.float64,
    )

    if not transitions:
        raise ValueError(
            "Cannot analyze an empty replay buffer."
        )

    terminal_mask = np.array(
        [transition.done for transition in transitions],
        dtype=bool,
    )
    non_terminal_mask = ~terminal_mask

    scaled_priorities = priorities**PER_ALPHA
    sampling_probabilities = (
        scaled_priorities / scaled_priorities.sum()
    )

    terminal_count = int(terminal_mask.sum())
    non_terminal_count = int(non_terminal_mask.sum())
    total_count = len(transitions)

    terminal_replay_share = terminal_count / total_count
    terminal_probability_share = float(
        sampling_probabilities[terminal_mask].sum()
    )

    print(
        f"Replay PER diagnostics - transitions: {total_count} "
        f"- terminal: {terminal_count} "
        f"- non-terminal: {non_terminal_count}"
    )

    if terminal_count > 0:
        terminal_priorities = priorities[terminal_mask]
        print(
            "Terminal priorities - "
            f"mean: {terminal_priorities.mean():.6f} - "
            f"median: {np.median(terminal_priorities):.6f}"
        )

    if non_terminal_count > 0:
        non_terminal_priorities = priorities[non_terminal_mask]
        print(
            "Non-terminal priorities - "
            f"mean: {non_terminal_priorities.mean():.6f} - "
            f"median: {np.median(non_terminal_priorities):.6f}"
        )

    if terminal_count > 0:
        oversampling_factor = (
            terminal_probability_share / terminal_replay_share
        )
        print(
            "Terminal PER sampling - "
            f"replay share: {terminal_replay_share:.6f} - "
            f"probability share: {terminal_probability_share:.6f} - "
            f"oversampling factor: {oversampling_factor:.3f}x"
        )

def analyze_terminal_per_categories(
    replay_buffer: ReplayBuffer,
    terminal_labels: dict[int, str],
) -> None:
    """Measure retained terminal categories under PER."""
    transitions = list(replay_buffer.buffer)
    priorities = np.asarray(
        list(replay_buffer.priorities),
        dtype=np.float64,
    )

    if not transitions or len(transitions) != len(priorities):
        raise ValueError("Invalid replay data.")

    probabilities = priorities**PER_ALPHA
    probabilities /= probabilities.sum()

    print("\nTerminal PER categories:")

    for category in ("truncated", "win", "loss", "draw"):
        indices = [
            index
            for index, transition in enumerate(transitions)
            if terminal_labels.get(id(transition)) == category
        ]

        if not indices:
            print(f"{category}: count=0")
            continue

        category_priorities = priorities[indices]
        replay_share = len(indices) / len(transitions)
        probability_share = float(probabilities[indices].sum())

        print(
            f"{category}: count={len(indices)} "
            f"- replay share={replay_share:.6f} "
            f"- priority mean={category_priorities.mean():.6f} "
            f"- priority median={np.median(category_priorities):.6f} "
            f"- PER share={probability_share:.6f} "
            f"- oversampling={probability_share / replay_share:.3f}x"
        )

def analyze_replay_diversity(
    replay_buffer: ReplayBuffer,
) -> None:
    """Summarize observable diversity in the retained replay."""
    transitions = list(replay_buffer.buffer)

    if not transitions:
        raise ValueError(
            "Cannot analyze an empty replay buffer."
        )

    action_counts = Counter(
        transition.action
        for transition in transitions
    )

    encoded_state_counts = Counter(
        transition.state.numpy().tobytes()
        for transition in transitions
    )

    state_action_counts = Counter(
        (
            transition.state.numpy().tobytes(),
            transition.action,
        )
        for transition in transitions
    )

    total = len(transitions)

    def top_action_share(count: int) -> float:
        return sum(
            frequency
            for _, frequency in action_counts.most_common(count)
        ) / total

    print(
        f"Replay diversity "
        f"- transitions: {total} "
        f"- unique actions: {len(action_counts)} "
        f"- unique encoded states: {len(encoded_state_counts)} "
        f"- unique state-action pairs: {len(state_action_counts)}"
    )

    print(
        f"Replay repetition "
        f"- repeated encoded states: "
        f"{total - len(encoded_state_counts)} "
        f"- repeated state-action pairs: "
        f"{total - len(state_action_counts)}"
    )

    print(
        "Action concentration "
        f"- top 10: {top_action_share(10):.3f} "
        f"- top 25: {top_action_share(25):.3f} "
        f"- top 50: {top_action_share(50):.3f}"
    )

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="State-Action DQN diagnostic training probe."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        choices=range(1, 100_001),
        metavar="1..100000",
    )
    parser.add_argument(
        "--architecture",
        choices=("spatial", "standard"),
        default="spatial",
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )

    parser.add_argument(
        "--initialization-seed",
        type=int,
        default=12345,
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=54321,
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=67890,
    )

    return parser.parse_args(args)

def print_probe_configuration(
    args,
    *,
    bootstrap_truncation,
    initialization_seed,
    evaluation_seed,
    training_seed,
    evaluation_episodes_per_color,
    target_update_frequency,
    replay_buffer,
    max_agent_steps,
    batch_size,
    min_replay_size,
):
    print("State-Action DQN probe")
    print(f"Device: {args.device}")
    print(
        f"Architecture: {args.architecture} "
        f"- bootstrap truncation: {bootstrap_truncation}"
    )
    print(
        f"Seeds - initialization: {initialization_seed} "
        f"- evaluation: {evaluation_seed} "
        f"- training: {training_seed}"
    )
    print(
        f"Evaluation games per color: {evaluation_episodes_per_color} "
        f"- target update frequency: {target_update_frequency} "
        f"- replay capacity: {replay_buffer.buffer.maxlen}"
    )
    print(
        f"Training episodes: {args.episodes} "
        f"- max agent steps: {max_agent_steps} "
        f"- batch size: {batch_size} "
        f"- min replay size: {min_replay_size}"
    )

class TrainingProfiler:
    """Measure time spent in action selection and neural updates."""

    def __init__(self):
        self.action_seconds = 0.0
        self.update_seconds = 0.0
        self.action_calls = 0
        self.update_calls = 0
        self.device = torch.device("cpu")

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def attach(self, agent):
        self.device = torch.device(
            getattr(agent, "device", "cpu")
        )
        original_select_action = agent.select_action
        original_train_step = agent.train_step

        def timed_select_action(*args, **kwargs):
            self.synchronize()
            start = perf_counter()
            try:
                return original_select_action(*args, **kwargs)
            finally:
                self.synchronize()
                self.action_seconds += perf_counter() - start
                self.action_calls += 1

        def timed_train_step(*args, **kwargs):
            self.synchronize()
            start = perf_counter()
            try:
                return original_train_step(*args, **kwargs)
            finally:
                self.synchronize()
                self.update_seconds += perf_counter() - start
                self.update_calls += 1

        agent.select_action = timed_select_action
        agent.train_step = timed_train_step

    def print_summary(self, total_seconds):
        other_seconds = max(
            0.0,
            total_seconds - self.action_seconds - self.update_seconds,
        )

        print("\nTraining performance profile")
        print(
            f"Action selection: {self.action_seconds:.2f}s "
            f"({self.action_calls} calls)"
        )
        print(
            f"Neural updates: {self.update_seconds:.2f}s "
            f"({self.update_calls} calls)"
        )
        print(f"Other training work: {other_seconds:.2f}s")
        print(f"Total training: {total_seconds:.2f}s")

def main() -> None:
    """
    Run a short CPU training probe for the State-Action DQN.

    The goal is to verify sustained end-to-end training and measure
    its wall-clock cost before deciding whether longer CPU training
    or explicit CUDA support should come next.
    """
    args = parse_args()
    training_episodes = args.episodes
    evaluation_episodes_per_color = 20
    max_agent_steps = 150
    batch_size = 32
    min_replay_size = 1000
    target_update_frequency = 10
    bootstrap_truncation=True


    initialization_seed = args.initialization_seed
    evaluation_seed = args.evaluation_seed
    training_seed = args.training_seed

    set_random_seed(initialization_seed)

    env = ChessEnv()

    agent = StateActionDQNAgent(
        epsilon=1.0,
        spatial=(args.architecture == "spatial"),
        device=args.device,
    )

    opponent = RandomAgent()

    replay_buffer = ReplayBuffer(
        capacity=10_000,
    )

    print_probe_configuration(
        args,
        bootstrap_truncation=bootstrap_truncation,
        initialization_seed=initialization_seed,
        evaluation_seed=evaluation_seed,
        training_seed=training_seed,
        evaluation_episodes_per_color=evaluation_episodes_per_color,
        target_update_frequency=target_update_frequency,
        replay_buffer=replay_buffer,
        max_agent_steps=max_agent_steps,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
    )

    print("\nInitial greedy evaluation...")

    set_random_seed(evaluation_seed)    
    initial_evaluation = evaluate_against_random_both_colors(
        env=env,
        agent=agent,
        opponent=opponent,
        episodes_per_color=evaluation_episodes_per_color,
        max_agent_steps=max_agent_steps,
    )

    initial_score = score_evaluation(
        initial_evaluation
    )

    print(
        f"Initial evaluation "
        f"- games: {initial_evaluation.episodes} "
        f"- wins: {initial_evaluation.wins} "
        f"- draws: {initial_evaluation.draws} "
        f"- losses: {initial_evaluation.losses} "
        f"- truncated: {initial_evaluation.truncated} "
        f"- score: {initial_score:.3f}"
    )

    print(
        f"Initial truncation diagnostics "
        f"- claimable threefold: "
        f"{initial_evaluation.truncated_claimable_threefold} "
        f"- claimable fifty-move: "
        f"{initial_evaluation.truncated_claimable_fifty_moves} "
        f"- without claimable draw: "
        f"{initial_evaluation.truncated_without_claimable_draw}"
    )

    print(
        f"Initial material diagnostics "
        f"- avg total material: "
        f"{initial_evaluation.truncated_average_total_material} "
        f"- avg material balance: "
        f"{initial_evaluation.truncated_average_material_balance} "
        f"- avg absolute material balance: "
        f"{initial_evaluation.truncated_average_absolute_material_balance}"
    )

    print("\nTraining...")

    set_random_seed(training_seed)

    terminal_tracker = TerminalTransitionTracker(
        replay_buffer=replay_buffer,
    )

    sample_observer = PrioritizedSampleObserver(replay_buffer)
    replay_buffer.sample_prioritized = sample_observer.sample_prioritized

    profiler = TrainingProfiler()
    profiler.attach(agent)
    profiler.synchronize()
    start_time = perf_counter()

    results = train_against_random(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=training_episodes,
        max_agent_steps=max_agent_steps,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        target_update_frequency=target_update_frequency,
        alternate_colors=True,
        progress_callback=terminal_tracker.record_episode,
        bootstrap_truncation=bootstrap_truncation,
    )

    final_terminal_labels = terminal_tracker.get_final_labels()

    print(
        "Final replay terminal categories:",
        dict(Counter(final_terminal_labels.values())),
    )
    profiler.synchronize()
    elapsed_seconds = (
        perf_counter() - start_time
    )

    profiler.print_summary(elapsed_seconds)

    summary = summarize_training(
        results
    )

    print(
        f"Training completed "
        f"- time: {elapsed_seconds:.2f}s "
        f"- seconds/episode: "
        f"{elapsed_seconds / training_episodes:.2f}"
    )

    print(
        f"Training summary "
        f"- wins: {summary.wins} "
        f"- draws: {summary.draws} "
        f"- losses: {summary.losses} "
        f"- truncated: {summary.truncated} "
        f"- avg plies: {summary.average_plies:.2f} "
        f"- avg reward: {summary.average_reward:.4f} "
        f"- avg loss: "
        f"{summary.average_loss if summary.average_loss is not None else 'N/A'} "
        f"- epsilon: {summary.final_epsilon:.4f} "
        f"- replay: {summary.replay_size}"
    )

    print(
        f"Truncation diagnostics "
        f"- claimable threefold: "
        f"{summary.truncated_claimable_threefold} "
        f"- claimable fifty-move: "
        f"{summary.truncated_claimable_fifty_moves} "
        f"- without claimable draw: "
        f"{summary.truncated_without_claimable_draw}"
    )

    print(
        f"Truncated material diagnostics "
        f"- avg total material: "
        f"{summary.truncated_average_total_material} "
        f"- avg material balance: "
        f"{summary.truncated_average_material_balance} "
        f"- avg absolute material balance: "
        f"{summary.truncated_average_absolute_material_balance}"
    )

    print("\nAnalyzing final replay learning signal...")

    analyze_replay_learning_signal(
        agent=agent,
        replay_buffer=replay_buffer,
    )

    analyze_replay_priorities(replay_buffer)

    analyze_terminal_per_categories(
        replay_buffer=replay_buffer,
        terminal_labels=final_terminal_labels,
    )

    analyze_observed_per_samples(
        observer=sample_observer,
        tracker=terminal_tracker,
    )    

    print("\nAnalyzing final replay diversity...")

    analyze_replay_diversity(
        replay_buffer=replay_buffer,
    )

    print("\nFinal greedy evaluation...")

    set_random_seed(evaluation_seed)

    final_evaluation = evaluate_against_random_both_colors(
        env=env,
        agent=agent,
        opponent=opponent,
        episodes_per_color=evaluation_episodes_per_color,
        max_agent_steps=max_agent_steps,
    )

    final_score = score_evaluation(
        final_evaluation
    )

    print(
        f"Final evaluation "
        f"- games: {final_evaluation.episodes} "
        f"- wins: {final_evaluation.wins} "
        f"- draws: {final_evaluation.draws} "
        f"- losses: {final_evaluation.losses} "
        f"- truncated: {final_evaluation.truncated} "
        f"- score: {final_score:.3f}"
    )

    print(
        f"Final truncation diagnostics "
        f"- claimable threefold: "
        f"{final_evaluation.truncated_claimable_threefold} "
        f"- claimable fifty-move: "
        f"{final_evaluation.truncated_claimable_fifty_moves} "
        f"- without claimable draw: "
        f"{final_evaluation.truncated_without_claimable_draw}"
    )

    print(
        f"Final material diagnostics "
        f"- avg total material: "
        f"{final_evaluation.truncated_average_total_material} "
        f"- avg material balance: "
        f"{final_evaluation.truncated_average_material_balance} "
        f"- avg absolute material balance: "
        f"{final_evaluation.truncated_average_absolute_material_balance}"
    )

    print("\nSaving final greedy diagnostic game...")

    set_random_seed(evaluation_seed)

    with TemporaryDirectory() as temporary_directory:
        diagnostic_path = Path(temporary_directory) / "diagnostic.pgn"
        diagnostic_buffer = ReplayBuffer(capacity=max_agent_steps)

        diagnostic_result = save_greedy_evaluation_game(
            env=env,
            agent=agent,
            opponent=opponent,
            path=diagnostic_path,
            max_agent_steps=max_agent_steps,
            agent_color=chess.WHITE,
            replay_buffer=diagnostic_buffer,
            bootstrap_truncation=bootstrap_truncation,
        )

        termination = diagnostic_result.final_info.get("termination")

        print(
            f"Diagnostic game - plies: {diagnostic_result.total_plies} "
            f"- truncated: {diagnostic_result.truncated} "
            f"- termination: {termination}"
        )

        print("\nFinal-policy diagnostic-game learning signal...")
        analyze_replay_learning_signal(
            agent=agent,
            replay_buffer=diagnostic_buffer,
        )

        print("\nFinal-policy diagnostic-game diversity...")
        analyze_replay_diversity(
            replay_buffer=diagnostic_buffer,
        )

        print("\nAnalyzing reconstructed greedy choices...")
        analyze_greedy_pgn(
            path=diagnostic_path,
            agent=agent,
            agent_color=chess.WHITE,
        )

        print_diagnostic_pgn(diagnostic_path)

if __name__ == "__main__":
    main()