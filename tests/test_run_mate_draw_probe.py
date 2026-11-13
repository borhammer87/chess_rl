import chess
import pytest
import torch

from scripts import run_mate_draw_probe as probe


@pytest.mark.parametrize("black", [False, True])
def test_all_terminal_actions_and_deterministic_pair(black):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    if black:
        board = board.mirror()
    pair = probe.find_terminal_pair(board)
    assert pair is not None
    for field, termination in (("mate_moves", chess.Termination.CHECKMATE),
                               ("stalemate_moves", chess.Termination.STALEMATE)):
        expected = []
        for move in board.legal_moves:
            end = board.copy()
            end.push(move)
            if end.outcome() and end.outcome().termination == termination:
                expected.append(move)
        assert pair[field] == sorted(expected, key=lambda move: move.uci())
        assert expected
    assert pair["mate_move"] == pair["mate_moves"][0]
    assert pair["stalemate_move"] == pair["stalemate_moves"][0]


def test_preserves_board_and_history():
    board = chess.Board("7k/8/6K1/8/8/8/5Q2/8 w - - 0 1")
    board.push_uci("f2f7")
    board.push(chess.Move.null())
    before = board.fen(), list(board.move_stack), board.root().fen()
    assert probe.find_terminal_pair(board) is not None
    assert (board.fen(), list(board.move_stack), board.root().fen()) == before


@pytest.mark.parametrize("fen", [chess.STARTING_FEN, "8/8/8/8/8/8/8/8 w - - 0 1",
                               "7k/8/5KQ1/8/8/8/8/8 b - - 0 1"])
def test_no_pair(fen):
    assert probe.find_terminal_pair(chess.Board(fen)) is None


def test_generation_reproducible_and_counts():
    def signature(pairs):
        return [(p["board"].fen(), p["mate_moves"], p["stalemate_moves"],
                 p["mate_move"], p["stalemate_move"]) for p in pairs]
    first, counts = probe.generate_terminal_pairs(20, seed=42)
    second, other = probe.generate_terminal_pairs(20, seed=42)
    assert signature(first) == signature(second)
    assert counts == other
    assert counts["candidates"] == 20
    assert counts["pairs"] == len(first) == counts["white"] + counts["black"]
    assert first
    for pair in first:
        board = pair["board"]
        assert board.is_valid() and len(board.piece_map()) == 3
        for field, check in (("mate_move", "is_checkmate"), ("stalemate_move", "is_stalemate")):
            end = board.copy()
            assert pair[field] in end.legal_moves
            end.push(pair[field])
            assert getattr(end, check)()


def test_existing_generator_called_once_and_empty_filter(monkeypatch, capsys):
    calls = []
    def generate(count, seed):
        calls.append((count, seed))
        return [chess.Board()]
    monkeypatch.setattr(probe, "generate_mate_positions", generate)
    probe.main(["--candidates", "1", "--seed", "7"])
    assert calls == [(1, 7)]
    output = capsys.readouterr().out
    assert "candidates=1 pairs=0 white=0 black=0" in output
    assert "No mate/stalemate pairs found" in output


def test_zero_and_negative_candidates():
    assert probe.generate_terminal_pairs(0) == ([], dict(candidates=0, pairs=0, white=0, black=0))
    with pytest.raises(ValueError, match="negative"):
        probe.generate_terminal_pairs(-1)


def symmetries(board):
    # Generate D4 independently: rotations from transpose + vertical reflection.
    for reflected in (board, board.transform(chess.flip_horizontal)):
        current = reflected
        for _ in range(4):
            yield current
            current = current.transform(chess.flip_diagonal).transform(chess.flip_vertical)


def test_key_invariant_under_symmetries_and_color_exchange():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    key = probe.canonical_kqk_key(board)
    variants = list(symmetries(board))
    assert len({b.board_fen() for b in variants}) == 8
    for variant in variants:
        assert probe.canonical_kqk_key(variant) == key
        swapped = chess.Board(None)
        swapped.turn = not variant.turn
        for square, piece in variant.piece_map().items():
            swapped.set_piece_at(square, chess.Piece(piece.piece_type, not piece.color))
        assert probe.canonical_kqk_key(swapped) == key


def test_split_preserves_records_actions_and_separates_families():
    pairs, _ = probe.generate_terminal_pairs(50, seed=12345)
    before = [(p["board"].fen(), list(p["board"].move_stack),
               p["mate_move"], p["stalemate_move"],
               list(p["mate_moves"]), list(p["stalemate_moves"])) for p in pairs]
    adjustment, evaluation = probe.split_terminal_pairs(pairs, seed=12345)
    assert adjustment and evaluation
    keys = lambda records: {probe.canonical_kqk_key(p["board"]) for p in records}
    assert keys(adjustment).isdisjoint(keys(evaluation))
    assert len(keys(evaluation)) == max(1, (len(keys(pairs)) + 2) // 5)
    assert sorted(map(id, adjustment + evaluation)) == sorted(map(id, pairs))
    again = probe.split_terminal_pairs(pairs, seed=12345)
    assert [list(map(id, group)) for group in again] == [list(map(id, adjustment)), list(map(id, evaluation))]
    reordered = probe.split_terminal_pairs(list(reversed(pairs)), seed=12345)
    assert keys(reordered[0]) == keys(adjustment)
    assert keys(reordered[1]) == keys(evaluation)
    assert before == [(p["board"].fen(), list(p["board"].move_stack),
                       p["mate_move"], p["stalemate_move"],
                       p["mate_moves"], p["stalemate_moves"]) for p in pairs]


def test_empty_and_single_family_have_no_evaluation(capsys):
    assert probe.split_terminal_pairs([]) == ([], [])
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    pairs = [probe.find_terminal_pair(b) for b in symmetries(board)]
    adjustment, evaluation = probe.split_terminal_pairs(pairs)
    assert list(map(id, adjustment)) == list(map(id, pairs))
    assert evaluation == []
    probe.main(["--candidates", "0"])
    assert "No family-disjoint evaluation possible" in capsys.readouterr().out


def test_two_families_each_get_a_partition():
    pairs, _ = probe.generate_terminal_pairs(30)
    distinct = {}
    for pair in pairs:
        distinct.setdefault(probe.canonical_kqk_key(pair["board"]), pair)
    adjustment, evaluation = probe.split_terminal_pairs(list(distinct.values())[:2])
    assert len(adjustment) == len(evaluation) == 1


def test_key_rejects_non_kqk():
    with pytest.raises(ValueError, match="KQK"):
        probe.canonical_kqk_key(chess.Board())


class FixedPolicy(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.child = torch.nn.Identity()
        self.scores, self.calls = scores, []

    def evaluate_action_ids(self, state, actions):
        assert not torch.is_grad_enabled() and not self.training
        assert state.device == self.weight.device
        self.calls.append(actions)
        return torch.tensor([self.scores.get(action, -2.) for action in actions])


@pytest.mark.parametrize("black", [False, True])
def test_fixed_pair_metrics_and_global_greedy(black):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    if black:
        board = board.mirror()
    pair = probe.find_terminal_pair(board)
    mate, draw = map(probe.encode_move, (pair["mate_move"], pair["stalemate_move"]))
    other = next(m for m in board.legal_moves if m not in pair["mate_moves"] and m != pair["stalemate_move"])
    policy = FixedPolicy({mate: .75, draw: .25, probe.encode_move(other): 2.})
    policy.child.eval()  # Preserve mixed submodule modes too.
    before = board.fen(), list(board.move_stack), policy.weight.clone()
    result = probe.evaluate_terminal_pairs([pair], policy)
    assert result == dict(positions=1, families=1, q_mate=.75, q_stalemate=.25,
                          mae_mate=.25, mae_stalemate=.25, margin=.5,
                          positive=1, negative=0, zero=0, greedy_mate=0,
                          positive_percent=100., negative_percent=0., zero_percent=0., greedy_mate_percent=0.)
    assert policy.calls == [[mate, draw], [probe.encode_move(m) for m in board.legal_moves]]
    assert policy.training and not policy.child.training
    assert policy.weight.grad is None and torch.equal(policy.weight, before[2])
    assert (board.fen(), list(board.move_stack)) == before[:2]
    another_mate = next(m for m in pair["mate_moves"] if m != pair["mate_move"])
    policy.scores[probe.encode_move(another_mate)] = 3.
    result = probe.evaluate_terminal_pairs([pair], policy)
    assert result["q_mate"] == .75 and result["greedy_mate"] == 1


def test_ties_and_empty_metrics():
    pair = probe.find_terminal_pair(chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"))
    policy = FixedPolicy({probe.encode_move(m): .5 for m in pair["board"].legal_moves})
    result = probe.evaluate_terminal_pairs([pair], policy)
    assert result["zero"] == 1 and result["zero_percent"] == 100
    assert result["greedy_mate"] == int(next(iter(pair["board"].legal_moves)) in pair["mate_moves"])
    empty = probe.evaluate_terminal_pairs([], policy)
    assert empty["positions"] == empty["families"] == 0
    assert empty["q_mate"] is None and empty["greedy_mate_percent"] is None


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("illegal", [False, True])
def test_mode_restored_on_error(training, illegal, monkeypatch):
    pair = probe.find_terminal_pair(chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"))
    policy = FixedPolicy({})
    policy.train(training)
    before = pair["board"].fen()
    if illegal:
        pair["mate_move"] = chess.Move.from_uci("a1a2")
    else:
        def fail(*args):
            raise ValueError("inference failed")
        monkeypatch.setattr(policy, "evaluate_action_ids", fail)
    with pytest.raises(ValueError):
        probe.evaluate_terminal_pairs([pair], policy)
    assert policy.training == training and pair["board"].fen() == before


def test_load_policy_without_rng_changes(tmp_path):
    import random
    import numpy as np
    with torch.device("meta"):
        net = probe.SpatialStateActionDQN()
    weights = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in net.state_dict().items()}
    path = tmp_path / "fixture.pt"
    torch.save({"metadata": {"architecture": "spatial"}, "agent": {"policy_net": weights}}, path)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    loaded = probe.load_policy(path)
    assert all(torch.equal(v, weights[k]) for k, v in loaded.state_dict().items())
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and np.array_equal(after[1], before[1][1]) and after[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    torch.save({"metadata": {"architecture": "other"}}, path)
    with pytest.raises(ValueError, match="spatial"):
        probe.load_policy(path)
    torch.save({"metadata": {"architecture": "spatial"}, "agent": {"policy_net": {}}}, path)
    with pytest.raises(RuntimeError):
        probe.load_policy(path)


def test_checkpoint_cli_optional_and_empty(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(probe, "load_policy", lambda path: calls.append(path) or FixedPolicy({}))
    probe.main(["--candidates", "0"])
    assert calls == []
    capsys.readouterr()
    probe.main(["--candidates", "0", "--checkpoint", "fixture.pt"])
    output = capsys.readouterr().out
    assert len(calls) == 1
    assert "checkpoint=fixture.pt" in output
    assert "q_mate=N/A" in output and "greedy_mate_percent=N/A" in output


@pytest.mark.parametrize("black", [False, True])
def test_terminal_batch_balanced_verified_and_preserves_history(black):
    board = chess.Board("7k/8/6K1/8/8/8/5Q2/8 w - - 0 1")
    board.push_uci("f2f7")
    board.push(chess.Move.null())
    if black:
        board = board.mirror()
    before = board.fen(), list(board.move_stack)
    pair = probe.find_terminal_pair(board)
    batch = probe.terminal_pair_batch([pair])
    assert [t.reward for t in batch] == [1., 0.]
    for transition, field in zip(batch, ("mate_move", "stalemate_move")):
        successor = board.copy(stack=True)
        successor.push(pair[field])
        assert transition.done and transition.next_legal_actions == []
        assert transition.action == probe.encode_move(pair[field])
        assert torch.equal(transition.state, probe.encode_board(board))
        assert torch.equal(transition.next_state, probe.encode_board(successor))
    assert (board.fen(), list(board.move_stack)) == before
    pair["mate_move"] = pair["stalemate_move"]
    with pytest.raises(ValueError, match="terminal result"):
        probe.terminal_pair_batch([pair])
    pair["mate_move"] = chess.Move.from_uci("a1a2")
    with pytest.raises(ValueError, match="legal"):
        probe.terminal_pair_batch([pair])


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_nested_equal(a, b)
    else:
        assert first == second


def test_adjustment_loads_full_adam_preserves_rng_and_never_saves(tmp_path, monkeypatch):
    import random
    import numpy as np
    agent = probe.StateActionDQNAgent(spatial=True, lr=.0002)
    pair = probe.find_terminal_pair(chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"))
    batch = probe.terminal_pair_batch([pair])
    agent.train_step(batch)  # Populate every Adam moment and step with synthetic data.
    state = agent.state_dict()
    path = tmp_path / "synthetic.pt"
    torch.save(dict(metadata=dict(architecture="spatial"), agent=state,
                    replay="must not restore", rng_state="must not restore"), path)
    original = path.read_bytes()
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected save or terminal bootstrap")
    monkeypatch.setattr(torch, "save", forbidden)
    restored = probe.load_adjustment_agent(path)
    assert_nested_equal(state["policy_net"], restored.policy_net.state_dict())
    assert_nested_equal(state["optimizer"], restored.optimizer.state_dict())
    assert restored.optimizer.state_dict()["state"]
    monkeypatch.setattr(restored.target_net, "evaluate_legal_action_maxes", forbidden)
    target_before = {k: v.clone() for k, v in restored.target_net.state_dict().items()}
    probe.run_adjustment(restored, [pair], [], 1)
    assert_nested_equal(target_before, restored.target_net.state_dict())
    assert any(not torch.equal(v, restored.policy_net.state_dict()[k])
               for k, v in state["policy_net"].items())
    assert path.read_bytes() == original
    assert random.getstate() == before[0]
    assert_nested_equal(np.random.get_state()[0], before[1][0])
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert np.random.get_state()[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


@pytest.mark.parametrize("updates, milestones", [(0, [0]), (10, [0, 10]), (11, [0, 10]), (100, [0, 10, 100])])
def test_fixed_budget_batch_isolation_and_measurements(updates, milestones, monkeypatch):
    from types import SimpleNamespace
    pairs, _ = probe.generate_terminal_pairs(20)
    adjustment, evaluation = probe.split_terminal_pairs(pairs)
    calls, measured = [], []
    policy = FixedPolicy({})
    policy.eval()
    agent = SimpleNamespace(policy_net=policy, train_step=lambda batch: calls.append(batch))
    def measure(subset, network):
        assert network is policy
        measured.append((len(calls), subset))
        return {"margin": -999}  # Bad evaluation never changes the budget.
    monkeypatch.setattr(probe, "evaluate_terminal_pairs", measure)
    results = probe.run_adjustment(agent, adjustment, evaluation, updates)
    assert list(results) == milestones and len(calls) == updates
    assert measured == [(step, subset) for step in milestones for subset in (adjustment, evaluation)]
    assert not policy.training
    if calls:
        assert all(batch is calls[0] for batch in calls)
        assert len(calls[0]) == 2 * len(adjustment)
        for pair, mate, draw in zip(adjustment, calls[0][::2], calls[0][1::2]):
            assert torch.equal(mate.state, probe.encode_board(pair["board"]))
            assert torch.equal(draw.state, mate.state)
            assert (mate.reward, draw.reward) == (1., 0.)


def test_adjustment_rejects_invalid_inputs_and_restores_mode(monkeypatch):
    from types import SimpleNamespace
    pair = probe.find_terminal_pair(chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"))
    policy = FixedPolicy({}).eval()
    def fail(*args):
        raise RuntimeError("update failed")
    agent = SimpleNamespace(policy_net=policy, train_step=fail)
    for adjustment, evaluation, updates, message in (([], [], 1, "at least"),
            ([pair], [pair], 1, "overlap"), ([pair], [], -1, "non-negative")):
        with pytest.raises(ValueError, match=message):
            probe.run_adjustment(agent, adjustment, evaluation, updates)
    with pytest.raises(RuntimeError, match="update failed"):
        probe.run_adjustment(agent, [pair], [], 1)
    assert not policy.training


def test_adjustment_cli_requires_explicit_budget_and_checkpoint(monkeypatch, capsys):
    for args in (["--updates", "-1"], ["--updates", "10"]):
        with pytest.raises(SystemExit):
            probe.main(args)
    sentinel = object()
    monkeypatch.setattr(probe, "load_adjustment_agent", lambda path: sentinel)
    calls = []
    def run(agent, adjustment, evaluation, updates):
        calls.append((agent, updates))
        return {0: {"adjustment": {"positions": len(adjustment)}}}
    monkeypatch.setattr(probe, "run_adjustment", run)
    probe.main(["--candidates", "20", "--checkpoint", "synthetic.pt", "--updates", "10"])
    assert calls == [(sentinel, 10)]
    assert "step=0 adjustment" in capsys.readouterr().out
