
import chess

from scripts.run_mate_in_one_probe import (
    MATE_POSITIONS,
    find_mate_in_one_moves,
    build_mate_transitions,
    train_mate_batch,
)

import torch

from chess_rl.utils.action_encoder import encode_move
from scripts.run_mate_in_one_probe import evaluate_mate_position

def test_finds_fools_mate():
    board = chess.Board()
    for move in ("f2f3", "e7e5", "g2g4"):
        board.push_uci(move)

    mates = find_mate_in_one_moves(board)

    assert chess.Move.from_uci("d8h4") in mates
    assert all(move in board.legal_moves for move in mates)


def test_initial_position_has_no_mate_in_one():
    board = chess.Board()

    assert find_mate_in_one_moves(board) == []

def test_mate_detection_does_not_modify_board():
    board = chess.Board()

    for move in ("f2f3", "e7e5", "g2g4"):
        board.push_uci(move)

    original_fen = board.fen()
    original_moves = list(board.move_stack)

    find_mate_in_one_moves(board)

    assert board.fen() == original_fen
    assert board.move_stack == original_moves


def test_all_mate_positions_have_legal_mate_in_one():
    assert len(MATE_POSITIONS) == 4

    colors = set()

    for name, fen in MATE_POSITIONS.items():
        board = chess.Board(fen)

        assert board.is_valid(), name
        assert not board.is_game_over(), name

        mates = find_mate_in_one_moves(board)

        assert mates, name
        assert all(move in board.legal_moves for move in mates)
        colors.add(board.turn)

    assert colors == {chess.WHITE, chess.BLACK}


def test_evaluate_mate_position_detects_greedy_checkmate():
    board = chess.Board(MATE_POSITIONS["fools_mate_black"])
    mating_move = chess.Move.from_uci("d8h4")

    class FakeNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def evaluate_action_ids(self, state, action_ids):
            return torch.tensor(
                [
                    1.0 if action == encode_move(mating_move) else 0.0
                    for action in action_ids
                ],
                device=state.device,
            )

    original_fen = board.fen()
    original_history = list(board.move_stack)

    (
        move,
        is_mate,
        best_q,
        q_gap,
        mate_q,
        non_mate_q,
        advantage,
    ) = evaluate_mate_position(board, FakeNetwork())

    assert move == mating_move
    assert is_mate is True
    assert best_q == 1.0
    assert q_gap == 1.0
    assert mate_q == 1.0
    assert non_mate_q == 0.0
    assert advantage == 1.0
    assert board.fen() == original_fen
    assert board.move_stack == original_history


def test_run_mate_in_one_probe_summarizes_results(monkeypatch, capsys):
    from scripts import run_mate_in_one_probe as probe

    outcomes = iter([True, False, True, False])
    visited_colors = []

    def fake_evaluate(board, network):
        visited_colors.append(board.turn)
        return (
            chess.Move.from_uci("a2a3"),
            next(outcomes),
            0.5,
            0.1,
            0.4,
            0.5,
            -0.1,
        )

    monkeypatch.setattr(probe, "evaluate_mate_position", fake_evaluate)

    network = object()
    correct, total = probe.run_mate_in_one_probe(network)

    output = capsys.readouterr().out

    assert (correct, total) == (2, 4)
    assert "Mate-in-one accuracy: 2/4 (50.0%)" in output
    assert "mate Q=0.4000" in output
    assert "best non-mate Q=0.5000" in output
    assert "mate advantage=-0.1000" in output
    assert output.count("PASS") == 2
    assert output.count("FAIL") == 2
    assert set(visited_colors) == {chess.WHITE, chess.BLACK}



def test_main_uses_untrained_spatial_agent(monkeypatch, capsys):
    from scripts import run_mate_in_one_probe as probe

    captured = {}

    class FakeNetwork:
        def eval(self):
            captured["eval_called"] = True

    class FakeAgent:
        def __init__(self, spatial, device):
            captured["spatial"] = spatial
            captured["device"] = device
            self.policy_net = FakeNetwork()

    def fake_run(network):
        captured["network"] = network
        return 0, 4

    monkeypatch.setattr(probe, "StateActionDQNAgent", FakeAgent)
    monkeypatch.setattr(probe, "run_mate_in_one_probe", fake_run)

    probe.main()

    assert captured["spatial"] is True
    assert captured["device"] == "cpu"
    assert captured["eval_called"] is True
    assert isinstance(captured["network"], FakeNetwork)
    assert "untrained - seed: 12345" in capsys.readouterr().out


def test_evaluate_mate_position_uses_network_device():
    board = chess.Board(MATE_POSITIONS["fools_mate_black"])

    class FakeNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def evaluate_action_ids(self, state, action_ids):
            assert state.device == self.weight.device
            return torch.zeros(len(action_ids), device=state.device)

    (
        move,
        is_mate,
        best_q,
        q_gap,
        mate_q,
        non_mate_q,
        advantage,
    ) = evaluate_mate_position(board, FakeNetwork())

    assert move in board.legal_moves
    assert isinstance(is_mate, bool)
    assert best_q == 0.0
    assert q_gap == 0.0
    assert mate_q == 0.0
    assert non_mate_q == 0.0
    assert advantage == 0.0

def test_build_mate_transitions_are_real_terminal_wins():
    for fen in MATE_POSITIONS.values():
        board = chess.Board(fen)
        original_fen = board.fen()

        transitions = build_mate_transitions(board)
        mate_moves = find_mate_in_one_moves(board)

        assert len(transitions) == len(mate_moves)
        assert transitions

        assert {t.action for t in transitions} == {
            encode_move(move) for move in mate_moves
        }

        for transition in transitions:
            assert transition.reward == 1.0
            assert transition.done is True
            assert transition.next_legal_actions == []
            assert transition.state.shape == (18, 8, 8)
            assert transition.next_state.shape == (18, 8, 8)

        assert board.fen() == original_fen

def test_train_mate_batch_uses_terminal_transitions():
    board = chess.Board(MATE_POSITIONS["fools_mate_black"])
    captured = {}

    class FakeAgent:
        def train_step(self, transitions):
            captured["transitions"] = transitions
            return 0.25

    loss = train_mate_batch(FakeAgent(), [board])

    assert loss == 0.25
    assert captured["transitions"]
    assert all(t.done for t in captured["transitions"])
    assert all(t.reward == 1.0 for t in captured["transitions"])
    assert all(not t.next_legal_actions for t in captured["transitions"])


def test_train_mate_batch_rejects_empty_positions():
    import pytest

    class FakeAgent:
        def train_step(self, transitions):
            raise AssertionError("Training must not be called.")

    with pytest.raises(ValueError, match="cannot be empty"):
        train_mate_batch(FakeAgent(), [])

def test_run_mate_learning_probe_trains_and_evaluates(
    monkeypatch, capsys
):
    from scripts import run_mate_in_one_probe as probe

    calls = {"training": 0, "evaluation": 0}

    class FakeAgent:
        def __init__(self, spatial, device):
            assert spatial is True
            assert device == "cpu"
            self.policy_net = object()

    def fake_train(agent, positions):
        calls["training"] += 1
        assert len(positions) == 3
        assert all(
            board.fen() != MATE_POSITIONS["fools_mate_white"]
            for board in positions
        )
        return 0.125

    def fake_evaluate(network, positions=None):
        calls["evaluation"] += 1
        assert positions is not None
        assert len(positions) in (1, 3)
        return (calls["evaluation"], len(positions))

    monkeypatch.setattr(probe, "StateActionDQNAgent", FakeAgent)
    monkeypatch.setattr(probe, "train_mate_batch", fake_train)
    monkeypatch.setattr(probe, "run_mate_in_one_probe", fake_evaluate)

    initial, final = probe.run_mate_learning_probe(
        seed=12345,
        updates=3,
    )

    assert initial == (2, 1)
    assert final == (4, 1)
    assert calls == {"training": 3, "evaluation": 4}

    output = capsys.readouterr().out
    assert "Training updates: 3" in output
    assert "Final loss: 0.125000" in output

def test_run_mate_in_one_probe_accepts_subset(monkeypatch):
    from scripts import run_mate_in_one_probe as probe

    visited = []

    def fake_evaluate(board, network):
        visited.append(board.fen())
        return chess.Move.from_uci("a2a3"), True, 1.0, 0.1, 1.0, 0.0, 1.0

    monkeypatch.setattr(probe, "evaluate_mate_position", fake_evaluate)

    subset = {"only": MATE_POSITIONS["fools_mate_black"]}
    assert probe.run_mate_in_one_probe(object(), subset) == (1, 1)
    assert visited == [
        chess.Board(subset["only"]).fen()
    ]

def test_build_mate_replay_contains_real_terminal_wins():
    from scripts.run_mate_in_one_probe import (
        build_mate_replay,
        build_mate_transitions,
    )

    positions = [
        chess.Board(MATE_POSITIONS["fools_mate_black"]),
        chess.Board(MATE_POSITIONS["fools_mate_white"]),
    ]

    replay = build_mate_replay(positions)

    expected = sum(
        len(build_mate_transitions(board))
        for board in positions
    )

    assert len(replay) == expected

    for transition in replay.buffer:
        assert transition.reward == 1.0
        assert transition.done is True
        assert transition.next_legal_actions == []


def test_build_mate_replay_rejects_empty_positions():
    import pytest
    from scripts.run_mate_in_one_probe import build_mate_replay

    with pytest.raises(ValueError, match="No mate transitions"):
        build_mate_replay([])

def test_train_mate_replay_uses_sampled_minibatches():
    from scripts.run_mate_in_one_probe import train_mate_replay

    class FakeReplay:
        def __len__(self):
            return 100

        def sample(self, batch_size):
            assert batch_size == 4
            return [object()] * batch_size

    class FakeAgent:
        def __init__(self):
            self.calls = 0

        def train_step(self, batch):
            assert len(batch) == 4
            self.calls += 1
            return float(self.calls)

    agent = FakeAgent()

    losses = train_mate_replay(
        agent,
        FakeReplay(),
        updates=3,
        batch_size=4,
    )

    assert losses == [1.0, 2.0, 3.0]
    assert agent.calls == 3


def test_train_mate_replay_rejects_invalid_parameters():
    import pytest
    from scripts.run_mate_in_one_probe import train_mate_replay

    class FakeReplay:
        def __len__(self):
            return 10

    with pytest.raises(ValueError, match="Updates must be positive"):
        train_mate_replay(object(), FakeReplay(), updates=0)

    with pytest.raises(ValueError, match="Invalid batch size"):
        train_mate_replay(
            object(), FakeReplay(), updates=1, batch_size=11
        )


def test_generated_mate_experiment_collects_attempts(
    monkeypatch, capsys
):
    from scripts import run_mate_in_one_probe as probe

    class FakeNetwork:
        def eval(self):
            pass

    class FakeAgent:
        def __init__(self, spatial, device):
            self.policy_net = FakeNetwork()
            self.epsilon = 1.0
            self.decays = 0

        def decay_epsilon(self):
            self.decays += 1
            self.epsilon *= 0.9

    calls = {"attempts": 0, "train": 0}

    def fake_collect(agent, board, replay):
        calls["attempts"] += 1
        replay.push(
            state=torch.zeros(12, 8, 8),
            action=0,
            reward=0.0,
            next_state=torch.zeros(12, 8, 8),
            done=True,
            next_legal_actions=[],
        )
        return 0.0

    def fake_train(agent, replay, updates, batch_size):
        assert updates == 1
        assert batch_size == 2
        assert len(replay) >= 2
        calls["train"] += 1
        return [0.25]

    monkeypatch.setattr(probe, "StateActionDQNAgent", FakeAgent)
    monkeypatch.setattr(probe, "collect_mate_attempt", fake_collect)
    monkeypatch.setattr(probe, "train_mate_replay", fake_train)
    monkeypatch.setattr(
        probe, "evaluate_mate_position",
        lambda board, network: (None, False, 0, 0, 0, 0, -1),
    )

    results = probe.run_generated_mate_experiment(
        train_count=4,
        test_count=2,
        updates=2,
        batch_size=2,
    )

    assert calls == {"attempts": 8, "train": 2}
    assert all(result[0] == 0 for result in results)
    assert "Attempts: 8 | Successes: 0" in capsys.readouterr().out


def test_collect_mate_attempt_rewards_checkmate():
    from scripts.run_mate_in_one_probe import collect_mate_attempt
    from chess_rl.utils.replay_buffer import ReplayBuffer

    board = chess.Board(MATE_POSITIONS["fools_mate_black"])
    original_fen = board.fen()
    mating_move = chess.Move.from_uci("d8h4")

    class FakeAgent:
        def select_action(self, state, legal_moves):
            assert mating_move in legal_moves
            return encode_move(mating_move)

    replay = ReplayBuffer(capacity=10)
    reward = collect_mate_attempt(FakeAgent(), board, replay)

    assert reward == 1.0
    assert len(replay) == 1
    assert replay.buffer[0].reward == 1.0
    assert replay.buffer[0].done is True
    assert replay.buffer[0].next_legal_actions == []
    assert board.fen() == original_fen


def test_collect_mate_attempt_records_failure():
    from scripts.run_mate_in_one_probe import collect_mate_attempt
    from chess_rl.utils.replay_buffer import ReplayBuffer

    board = chess.Board(MATE_POSITIONS["fools_mate_black"])
    wrong_move = chess.Move.from_uci("a7a6")

    class FakeAgent:
        def select_action(self, state, legal_moves):
            assert wrong_move in legal_moves
            return encode_move(wrong_move)

    replay = ReplayBuffer(capacity=10)
    reward = collect_mate_attempt(FakeAgent(), board, replay)

    assert reward == 0.0
    assert len(replay) == 1
    assert replay.buffer[0].reward == 0.0
    assert replay.buffer[0].done is True
    assert replay.buffer[0].next_legal_actions == []
