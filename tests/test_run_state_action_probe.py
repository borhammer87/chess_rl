from types import SimpleNamespace

import chess
import pytest
import torch

from scripts.run_state_action_probe import TerminalTransitionTracker
from chess_rl.utils.replay_buffer import ReplayBuffer


def add_transition(buffer, *, done=True):
    state = torch.zeros((18, 8, 8))

    buffer.push(
        state=state,
        action=0,
        reward=0.0,
        next_state=state,
        done=done,
        next_legal_actions=[],
    )


def episode_result(*, truncated=False, result="1-0"):
    return SimpleNamespace(
        truncated=truncated,
        done=not truncated,
        final_info={"result": result},
        agent_color=chess.WHITE,
    )


def test_tracker_classifies_real_win_and_truncation():
    buffer = ReplayBuffer(capacity=10)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer)
    tracker.record_episode(
        1, 2, episode_result(result="1-0")
    )

    add_transition(buffer)
    tracker.record_episode(
        2, 2, episode_result(truncated=True)
    )

    assert list(tracker.get_final_labels().values()) == [
        "win",
        "truncated",
    ]


def test_tracker_ignores_evicted_transitions():
    buffer = ReplayBuffer(capacity=1)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer)
    tracker.record_episode(
        1, 2, episode_result(result="1-0")
    )

    add_transition(buffer)
    tracker.record_episode(
        2, 2, episode_result(result="0-1")
    )

    assert list(tracker.get_final_labels().values()) == [
        "loss",
    ]


def test_tracker_rejects_unlabelled_terminal_transition():
    buffer = ReplayBuffer(capacity=2)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer)

    with pytest.raises(ValueError, match="Unlabelled"):
        tracker.get_final_labels()


def test_tracker_rejects_nonterminal_last_transition():
    buffer = ReplayBuffer(capacity=2)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer, done=False)

    with pytest.raises(ValueError, match="neither terminal"):
        tracker.record_episode(
            1, 1, episode_result()
        )


def test_tracker_labels_bootstrapped_truncation():
    buffer = ReplayBuffer(capacity=3)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer, done=False)

    tracker.record_episode(
        1,
        1,
        episode_result(truncated=True),
    )

    transition = buffer.buffer[-1]
    labels = tracker.get_final_labels()

    assert transition.done is False
    assert labels == {id(transition): "truncated"}


def test_terminal_per_categories(capsys):
    from scripts.run_state_action_probe import (
        analyze_terminal_per_categories,
    )

    buffer = ReplayBuffer(capacity=4)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer)
    tracker.record_episode(1, 2, episode_result(result="1-0"))

    add_transition(buffer)
    tracker.record_episode(2, 2, episode_result(truncated=True))

    analyze_terminal_per_categories(
        buffer,
        tracker.get_final_labels(),
    )

    output = capsys.readouterr().out

    assert "win: count=1" in output
    assert "truncated: count=1" in output
    assert "loss: count=0" in output
    assert "draw: count=0" in output
    assert "oversampling=1.000x" in output

def test_per_categories_include_bootstrapped_truncation(capsys):
    from scripts.run_state_action_probe import (
        analyze_terminal_per_categories,
    )

    buffer = ReplayBuffer(capacity=2)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer, done=False)
    tracker.record_episode(
        1, 1, episode_result(truncated=True)
    )

    analyze_terminal_per_categories(
        buffer, tracker.get_final_labels()
    )

    output = capsys.readouterr().out
    assert "truncated: count=1" in output
    assert "PER share=1.000000" in output

def test_terminal_per_categories_with_unequal_priorities(capsys):
    from scripts.run_state_action_probe import (
        analyze_terminal_per_categories,
    )
    from chess_rl.training.episodes import PER_ALPHA

    buffer = ReplayBuffer(capacity=3)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer)
    tracker.record_episode(1, 2, episode_result(result="1-0"))

    add_transition(buffer)
    tracker.record_episode(2, 2, episode_result(truncated=True))

    add_transition(buffer, done=False)

    buffer.update_priorities(
        indices=[0, 1, 2],
        priorities=[1.0, 4.0, 9.0],
    )

    analyze_terminal_per_categories(
        buffer,
        tracker.get_final_labels(),
    )

    output = capsys.readouterr().out

    total = 1.0**PER_ALPHA + 4.0**PER_ALPHA + 9.0**PER_ALPHA
    win_share = 1.0**PER_ALPHA / total
    truncated_share = 4.0**PER_ALPHA / total

    assert (
        f"PER share={win_share:.6f}" in output
    )
    assert (
        f"PER share={truncated_share:.6f}" in output
    )
    assert (
        f"oversampling={3 * win_share:.3f}x" in output
    )
    assert (
        f"oversampling={3 * truncated_share:.3f}x" in output
    )


def test_sample_observer_preserves_original_result(monkeypatch):
    from scripts.run_state_action_probe import (
        PrioritizedSampleObserver,
    )

    buffer = ReplayBuffer(capacity=3)
    add_transition(buffer, done=False)
    add_transition(buffer, done=True)

    original_transition = buffer.buffer[1]
    expected_indices = [1, 1]
    expected_weights = torch.tensor([0.5, 0.5])
    calls = []

    def fake_sample(*, batch_size, alpha, beta):
        calls.append((batch_size, alpha, beta))
        return (
            [original_transition, original_transition],
            expected_indices,
            expected_weights,
        )

    monkeypatch.setattr(
        buffer,
        "sample_prioritized",
        fake_sample,
    )

    observer = PrioritizedSampleObserver(buffer)

    result = observer.sample_prioritized(
        batch_size=2,
        alpha=0.6,
        beta=0.4,
    )

    assert calls == [(2, 0.6, 0.4)]
    assert result[0][0] is original_transition
    assert result[0][1] is original_transition
    assert result[1] is expected_indices
    assert result[2] is expected_weights
    assert observer.sampled_transitions == [
        original_transition,
        original_transition,
    ]


def test_sample_observer_does_not_sample_until_called(monkeypatch):
    from scripts.run_state_action_probe import (
        PrioritizedSampleObserver,
    )

    buffer = ReplayBuffer(capacity=2)
    calls = []

    def fake_sample(*, batch_size, alpha, beta):
        calls.append((batch_size, alpha, beta))
        return ([], [], torch.tensor([]))

    monkeypatch.setattr(
        buffer,
        "sample_prioritized",
        fake_sample,
    )

    observer = PrioritizedSampleObserver(buffer)

    assert calls == []
    assert observer.sampled_transitions == []

    observer.sample_prioritized(
        batch_size=1,
        alpha=0.6,
        beta=0.4,
    )

    assert len(calls) == 1



def test_observer_records_real_replay_sampling():
    from scripts.run_state_action_probe import (
        PrioritizedSampleObserver,
    )

    buffer = ReplayBuffer(capacity=3)
    add_transition(buffer, done=False)
    add_transition(buffer, done=True)

    observer = PrioritizedSampleObserver(buffer)
    buffer.sample_prioritized = observer.sample_prioritized

    transitions, indices, weights = buffer.sample_prioritized(
        batch_size=4,
        alpha=0.6,
        beta=0.4,
    )

    assert len(transitions) == 4
    assert len(indices) == 4
    assert len(weights) == 4
    assert len(observer.sampled_transitions) == 4

    for recorded, sampled in zip(
        observer.sampled_transitions,
        transitions,
    ):
        assert recorded is sampled


def test_observed_per_samples_include_evicted_terminals(capsys):
    from scripts.run_state_action_probe import (
        PrioritizedSampleObserver,
        analyze_observed_per_samples,
    )

    buffer = ReplayBuffer(capacity=1)
    tracker = TerminalTransitionTracker(buffer)
    observer = PrioritizedSampleObserver(buffer)

    add_transition(buffer)
    tracker.record_episode(1, 2, episode_result(result="1-0"))
    winning_transition = buffer.buffer[-1]

    observer.sampled_transitions.extend(
        [winning_transition, winning_transition]
    )

    add_transition(buffer, done=False)
    observer.sampled_transitions.append(buffer.buffer[-1])

    assert all(
        transition is not winning_transition
        for transition in buffer.buffer
    )

    analyze_observed_per_samples(observer, tracker)
    output = capsys.readouterr().out

    assert "total: 3" in output
    assert "win: count=2 - share=0.666667" in output
    assert "non-terminal: count=1 - share=0.333333" in output
    assert "truncated: count=0" in output

def test_observed_samples_include_bootstrapped_truncation(capsys):
    from scripts.run_state_action_probe import (
        PrioritizedSampleObserver,
        analyze_observed_per_samples,
    )

    buffer = ReplayBuffer(capacity=1)
    tracker = TerminalTransitionTracker(buffer)
    observer = PrioritizedSampleObserver(buffer)

    add_transition(buffer, done=False)
    tracker.record_episode(
        1, 1, episode_result(truncated=True)
    )

    truncated_transition = buffer.buffer[-1]
    observer.sampled_transitions.append(truncated_transition)

    add_transition(buffer, done=False)

    analyze_observed_per_samples(observer, tracker)

    output = capsys.readouterr().out
    assert "truncated: count=1 - share=1.000000" in output
    assert "non-terminal: count=0" in output

def test_observed_per_samples_without_selections(capsys):
    from scripts.run_state_action_probe import (
        PrioritizedSampleObserver,
        analyze_observed_per_samples,
    )

    buffer = ReplayBuffer(capacity=2)
    tracker = TerminalTransitionTracker(buffer)
    observer = PrioritizedSampleObserver(buffer)

    analyze_observed_per_samples(observer, tracker)

    output = capsys.readouterr().out

    assert "total: 0" in output
    assert "win: count=0 - share=0.000000" in output
    assert "non-terminal: count=0 - share=0.000000" in output

def test_probe_cli_defaults():
    from scripts.run_state_action_probe import parse_args

    args = parse_args([])

    assert args.episodes == 100
    assert args.architecture == "spatial"
    assert args.initialization_seed == 12345
    assert args.evaluation_seed == 54321
    assert args.training_seed == 67890
    assert args.device == "cpu"

def test_probe_cli_accepts_standard_architecture():
    from scripts.run_state_action_probe import parse_args

    args = parse_args([
        "--episodes", "5",
        "--architecture", "standard",
    ])

    assert args.episodes == 5
    assert args.architecture == "standard"

def test_probe_cli_accepts_cuda_device():
    from scripts.run_state_action_probe import parse_args

    args = parse_args(["--device", "cuda"])

    assert args.device == "cuda"


def test_probe_cli_rejects_invalid_device():
    from scripts.run_state_action_probe import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--device", "invalid"])

def test_probe_cli_rejects_invalid_episodes():
    from scripts.run_state_action_probe import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--episodes", "0"])

def test_probe_prints_effective_configuration(capsys):
    from scripts.run_state_action_probe import (
        parse_args,
        print_probe_configuration,
    )

    args = parse_args([
        "--episodes", "5",
        "--architecture", "standard",
    ])

    print_probe_configuration(
        args,
        bootstrap_truncation=True,
        initialization_seed=12345,
        evaluation_seed=54321,
        training_seed=67890,
        evaluation_episodes_per_color=20,
        target_update_frequency=10,
        replay_buffer=ReplayBuffer(capacity=10000),
        max_agent_steps=150,
        batch_size=32,
        min_replay_size=1000,
    )

    output = capsys.readouterr().out

    assert "Architecture: standard" in output
    assert "Training episodes: 5" in output
    assert "replay capacity: 10000" in output
    assert "initialization: 12345" in output
    assert "bootstrap truncation: True" in output
    assert "Device: cpu" in output

def test_probe_cli_accepts_custom_seeds():
    from scripts.run_state_action_probe import parse_args

    args = parse_args([
        "--initialization-seed", "101",
        "--evaluation-seed", "202",
        "--training-seed", "303",
    ])

    assert args.initialization_seed == 101
    assert args.evaluation_seed == 202
    assert args.training_seed == 303

def test_training_profiler_preserves_calls_and_reports_time(capsys):
    from scripts.run_state_action_probe import TrainingProfiler

    class FakeAgent:
        def select_action(self, state, legal_moves):
            return 42

        def train_step(self, batch, *, return_td_errors=False):
            return (0.5, [0.1]) if return_td_errors else 0.5

    agent = FakeAgent()
    profiler = TrainingProfiler()
    profiler.attach(agent)

    assert agent.select_action("state", ["move"]) == 42
    assert agent.train_step(
        ["transition"],
        return_td_errors=True,
    ) == (0.5, [0.1])

    assert profiler.action_calls == 1
    assert profiler.update_calls == 1
    assert profiler.action_seconds >= 0
    assert profiler.update_seconds >= 0

    profiler.print_summary(10.0)

    output = capsys.readouterr().out

    assert "Training performance profile" in output
    assert "Action selection:" in output
    assert "Neural updates:" in output
    assert "Other training work:" in output
    assert "Total training: 10.00s" in output

def test_replay_learning_signal_uses_agent_device(capsys):
    from scripts.run_state_action_probe import (
        analyze_replay_learning_signal,
    )
    from chess_rl.agents.state_action_dqn_agent import (
        StateActionDQNAgent,
    )
    from chess_rl.utils.action_encoder import encode_move

    agent = StateActionDQNAgent(spatial=True)
    buffer = ReplayBuffer(capacity=2)

    board = chess.Board()
    state = torch.zeros((18, 8, 8))

    buffer.push(
        state=state,
        action=encode_move(chess.Move.from_uci("e2e4")),
        reward=1.0,
        next_state=state,
        done=True,
        next_legal_actions=[],
    )

    analyze_replay_learning_signal(
        agent=agent,
        replay_buffer=buffer,
    )

    output = capsys.readouterr().out
    assert "TD" in output

def test_probe_main_passes_device_to_agent(monkeypatch):
    from scripts import run_state_action_probe as probe

    class StopAfterAgentCreation(Exception):
        pass

    def fake_agent(*args, **kwargs):
        assert kwargs["device"] == "cpu"
        raise StopAfterAgentCreation

    monkeypatch.setattr(
        probe,
        "StateActionDQNAgent",
        fake_agent,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["run_state_action_probe.py", "--episodes", "1"],
    )

    with pytest.raises(StopAfterAgentCreation):
        probe.main()

def test_greedy_pgn_analysis_uses_agent_device(
    tmp_path, monkeypatch
):
    from scripts import run_state_action_probe as probe
    from chess_rl.utils.action_encoder import encode_move

    move = chess.Move.from_uci("e2e4")
    game = chess.pgn.Game()
    game.add_variation(move)

    path = tmp_path / "game.pgn"
    path.write_text(str(game), encoding="utf-8")

    device = torch.device("meta")
    checked = []

    class FakeNetwork:
        def evaluate_action_ids(self, state, action_ids):
            assert state.device == device
            checked.append("q_values")
            return torch.zeros(len(action_ids))

    def fake_greedy_choice(*, network, state, legal_moves):
        assert state.device == device
        checked.append("greedy")
        return encode_move(move), 0.0, 0.0

    monkeypatch.setattr(
        probe,
        "evaluate_state_action_greedy_choice",
        fake_greedy_choice,
    )

    agent = SimpleNamespace(
        device=device,
        policy_net=FakeNetwork(),
    )

    probe.analyze_greedy_pgn(
        path=path,
        agent=agent,
        agent_color=chess.WHITE,
    )

    assert checked == ["greedy", "q_values"]

def test_training_profiler_synchronizes_cuda(monkeypatch):
    from scripts.run_state_action_probe import TrainingProfiler

    calls = []

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: calls.append("sync"),
    )

    agent = SimpleNamespace(
        device=torch.device("cuda"),
        select_action=lambda *args, **kwargs: calls.append("action"),
        train_step=lambda *args, **kwargs: calls.append("update"),
    )

    profiler = TrainingProfiler()
    profiler.attach(agent)

    agent.select_action(None, [])
    agent.train_step([])

    assert calls == [
        "sync", "action", "sync",
        "sync", "update", "sync",
    ]
    assert profiler.action_calls == 1
    assert profiler.update_calls == 1

def test_print_diagnostic_pgn_includes_complete_game(tmp_path, capsys):
    from scripts.run_state_action_probe import print_diagnostic_pgn

    path = tmp_path / "game.pgn"
    game_text = (
        '[Event "Diagnostic"]\n'
        '[Result "1/2-1/2"]\n\n'
        '1. Nf3 Nf6 1/2-1/2\n'
    )
    path.write_text(game_text, encoding="utf-8")

    print_diagnostic_pgn(path)

    output = capsys.readouterr().out
    assert "--- BEGIN DIAGNOSTIC PGN ---" in output
    assert game_text.strip() in output
    assert "--- END DIAGNOSTIC PGN ---" in output


def test_print_diagnostic_pgn_supports_all_results(tmp_path, capsys):
    from scripts.run_state_action_probe import print_diagnostic_pgn

    path = tmp_path / "game.pgn"

    for result in ("1-0", "0-1", "1/2-1/2", "*"):
        path.write_text(
            f'[Result "{result}"]\n\n{result}\n',
            encoding="utf-8",
        )

        print_diagnostic_pgn(path)

        output = capsys.readouterr().out
        assert f'[Result "{result}"]' in output

def test_diagnostic_termination_is_available_in_episode_result():
    result = episode_result(result="1/2-1/2")
    result.final_info["termination"] = "FIVEFOLD_REPETITION"

    assert result.final_info["termination"] == "FIVEFOLD_REPETITION"



def test_evaluate_mate_in_one_uses_existing_policy(monkeypatch, capsys):
    from scripts import run_state_action_probe as probe

    network = object()
    agent = SimpleNamespace(policy_net=network)
    received = []

    def fake_probe(passed_network):
        received.append(passed_network)
        return 2, 4

    monkeypatch.setattr(probe, "run_mate_in_one_probe", fake_probe)

    result = probe.evaluate_mate_in_one(agent, "Initial")

    assert result == (2, 4)
    assert received == [network]
    assert "Initial mate-in-one evaluation..." in capsys.readouterr().out


def test_main_evaluates_mates_before_and_after_training(monkeypatch):
    from scripts import run_state_action_probe as probe

    calls = []
    original_evaluate = probe.evaluate_mate_in_one

    def fake_evaluate(agent, label):
        calls.append((agent, label))
        return 0, 4

    monkeypatch.setattr(probe, "evaluate_mate_in_one", fake_evaluate)

    # Verificamos la integración en main sin ejecutar el
    # entrenamiento completo en este test.
    import inspect

    source = inspect.getsource(probe.main)

    assert source.index('evaluate_mate_in_one(agent, "Initial")') < (
        source.index("train_against_random(")
    )
    assert source.index('evaluate_mate_in_one(agent, "Final")') > (
        source.index("train_against_random(")
    )
