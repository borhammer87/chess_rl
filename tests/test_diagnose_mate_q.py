"""Regression oracle frozen from the probe before helper extraction."""
import chess
import pytest
import torch

from chess_rl.utils.action_encoder import decode_legal_action, encode_move
from chess_rl.utils.action_selection import evaluate_state_action_greedy_choice
from chess_rl.utils.board_encoder import encode_board
from scripts.run_mate_in_one_probe import (
    MATE_POSITIONS, evaluate_mate_actions, evaluate_mate_position,
    find_mate_in_one_moves,
)


def legacy_evaluate_mate_position(board, network):
    """Evaluate whether the network chooses an immediate checkmate."""
    legal_moves = list(board.legal_moves)
    state = encode_board(board).to(next(network.parameters()).device)

    action, best_q, q_gap = evaluate_state_action_greedy_choice(
        network=network,
        state=state,
        legal_moves=legal_moves,
    )

    chosen_move = decode_legal_action(action, legal_moves)
    mate_moves = find_mate_in_one_moves(board)

    with torch.no_grad():
        scores = network.evaluate_action_ids(
            state,
            [encode_move(move) for move in legal_moves],
        )

    mate_set = set(mate_moves)

    mate_q = max(
        (float(scores[i].item())
         for i, move in enumerate(legal_moves)
         if move in mate_set),
        default=None,
    )

    non_mate_q = max(
        (float(scores[i].item())
         for i, move in enumerate(legal_moves)
         if move not in mate_set),
        default=None,
    )

    mate_advantage = (
        mate_q - non_mate_q
        if mate_q is not None and non_mate_q is not None
        else None
    )

    return (
        chosen_move,
        chosen_move in mate_set,
        best_q,
        q_gap,
        mate_q,
        non_mate_q,
        mate_advantage,
    )


class FixedScores(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.values = values
        self.calls = []

    def evaluate_action_ids(self, state, action_ids):
        assert not torch.is_grad_enabled()
        self.calls.append(list(action_ids))
        return torch.tensor([self.values[a] for a in action_ids], device=state.device)


@pytest.mark.parametrize("fen", [*MATE_POSITIONS.values(), chess.STARTING_FEN])
@pytest.mark.parametrize("tied", [False, True])
def test_probe_matches_legacy_order_selection_and_numbers(fen, tied):
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    ids = [encode_move(move) for move in moves]
    values = {a: 0.75 if tied else (i % 7) * 0.13 - 0.4 for i, a in enumerate(ids)}
    old, new = FixedScores(values), FixedScores(values)
    before = board.fen(), list(board.move_stack)
    expected = legacy_evaluate_mate_position(board, old)
    actual = evaluate_mate_position(board, new)
    assert actual[:2] == expected[:2]
    for observed, reference in zip(actual[2:], expected[2:]):
        if reference is None:
            assert observed is None
        else:
            assert observed == pytest.approx(reference, rel=1e-6, abs=1e-7)
    assert new.calls == old.calls == [ids, ids]
    assert (board.fen(), list(board.move_stack)) == before


def test_zero_margin_can_select_non_mate():
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4"):
        board.push_uci(uci)
    moves = list(board.legal_moves)
    assert moves[0] not in find_mate_in_one_moves(board)
    network = FixedScores({encode_move(m): 0.75 for m in moves})
    before = board.fen(), list(board.move_stack)
    result = evaluate_mate_position(board, network)
    assert result == (moves[0], False, 0.75, 0.0, 0.75, 0.75, 0.0)
    details = evaluate_mate_actions(board, network)
    mates = find_mate_in_one_moves(board)
    assert details[4] == (mates[0], 0.75)
    assert details[5] == (moves[0], 0.75)
    assert (board.fen(), list(board.move_stack)) == before


from scripts.diagnose_mate_q import analyze_position, move_outcome


def capture_mate_board(black=False):
    board = chess.Board()
    for san in ("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"):
        board.push_san(san)
    return board.mirror() if black else board


@pytest.mark.parametrize("black", [False, True])
def test_position_errors_and_capture_mate_reward_for_both_colors(black):
    board = capture_mate_board(black)
    mate = chess.Move.from_uci("h4f2" if black else "h5f7")
    alternative = next(m for m in board.legal_moves if m not in find_mate_in_one_moves(board))
    values = {encode_move(m): -0.5 for m in board.legal_moves}
    values[encode_move(mate)] = 0.8
    values[encode_move(alternative)] = 0.9
    result = analyze_position(board, FixedScores(values))
    assert result["mate_move"] == mate.uci()
    assert result["alternative_move"] == alternative.uci()
    assert result["mate_selected"] is False
    assert result["mate_q"] == pytest.approx(0.8)
    assert result["alternative_q"] == pytest.approx(0.9)
    assert result["q_margin"] == pytest.approx(-0.1)
    assert result["seeded_target"] == 1.0
    assert result["seeded_error"] == pytest.approx(-0.2)
    assert result["seeded_absolute_error"] == pytest.approx(0.2)
    assert result["game_terminal_reward"] == 1.0
    assert result["game_material_reward"] == pytest.approx(0.01)
    assert result["game_mate_reward"] == pytest.approx(1.01)
    category, _, raw_reward, info = move_outcome(board, mate)
    assert category == "other_terminal"
    assert info["termination"] == "CHECKMATE"
    assert raw_reward == (-1.0 if black else 1.0)


@pytest.mark.parametrize("alternative,category,termination", [
    ("f7e6", "terminal_draw", "STALEMATE"),
    ("f7f6", "non_terminal", None),
])
def test_alternative_classification(alternative, category, termination):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    values = {encode_move(m): 0.0 for m in board.legal_moves}
    values[encode_move(chess.Move.from_uci(alternative))] = 2.0
    result = analyze_position(board, FixedScores(values))
    assert result["alternative_move"] == alternative
    assert result["alternative_category"] == category
    assert result["alternative_termination"] == termination


def test_outcome_preserves_history_for_fivefold_draw():
    board = chess.Board()
    cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
    for uci in cycle * 3 + cycle[:3]:
        board.push_uci(uci)
    before = board.fen(), list(board.move_stack)
    category, next_board, reward, info = move_outcome(board, chess.Move.from_uci("f6g8"))
    assert reward == 0.0
    assert next_board.move_stack == board.move_stack + [chess.Move.from_uci("f6g8")]
    assert next_board.is_game_over()
    assert category == "terminal_draw"
    assert info["termination"] == "FIVEFOLD_REPETITION"
    assert (board.fen(), list(board.move_stack)) == before


def test_missing_alternative_is_explicit(monkeypatch):
    # Isolate the record boundary: a helper result with no non-mate group.
    board = capture_mate_board()
    mate = find_mate_in_one_moves(board)[0]
    monkeypatch.setattr(
        "scripts.diagnose_mate_q.evaluate_mate_actions",
        lambda board, network: (mate, True, 1.2, None, (mate, 1.2), None),
    )
    result = analyze_position(board, FixedScores({}))
    assert result["alternative_category"] == "no_alternative"
    assert result["alternative_move"] is None
    assert result["alternative_q"] is None
    assert result["q_margin"] is None
    assert result["seeded_error"] == pytest.approx(0.2)


def test_analysis_preserves_board_weights_and_gradients():
    from chess_rl.models.spatial_state_action_dqn import SpatialStateActionDQN

    board = capture_mate_board()
    network = SpatialStateActionDQN()
    before_board = board.fen(), list(board.move_stack)
    before_weights = {k: v.clone() for k, v in network.state_dict().items()}
    assert analyze_position(board, network) is not None
    assert (board.fen(), list(board.move_stack)) == before_board
    for key, value in network.state_dict().items():
        assert torch.equal(value, before_weights[key])
    assert all(p.grad is None for p in network.parameters())


def test_analysis_skips_positions_without_mate():
    board = chess.Board()
    network = FixedScores({encode_move(m): 0.0 for m in board.legal_moves})
    assert analyze_position(board, network) is None


def test_analysis_skips_finished_games():
    board = capture_mate_board()
    board.push_uci("h5f7")
    network = FixedScores({})
    assert analyze_position(board, network) is None
    assert network.calls == []


from scripts.diagnose_mate_q import analyze_sets, export_csv, main, print_summary


def test_sets_deduplicate_and_detect_overlap():
    board = capture_mate_board()
    duplicate = board.copy()
    duplicate.fullmove_number += 20
    network = FixedScores({encode_move(m): 0.75 for m in board.legal_moves})
    records, counts, overlap = analyze_sets(network, [board, duplicate], [board.copy()])
    assert overlap == 1
    assert counts["seeded"] == dict(total=2, duplicates=1, skipped=0, analyzed=1)
    assert counts["games"]["analyzed"] == 1
    assert len(records) == 2
    assert all(r["overlap"] for r in records)
    assert records[0]["fen"] == board.fen()


def test_summary_and_csv_keep_missing_alternatives(tmp_path, capsys):
    row = dict(dataset="seeded", fen="example", overlap=False,
               alternative_category="no_alternative", mate_selected=True,
               mate_q=1.0, alternative_q=None, q_margin=None,
               seeded_error=0.0, seeded_absolute_error=0.0, game_mate_reward=1.0)
    counts = {"seeded": dict(total=1, duplicates=0, skipped=0, analyzed=1)}
    print_summary([row], counts, 0)
    output = capsys.readouterr().out
    assert "no_alternative: n=1" in output
    assert "Qalt=N/A(n=0)" in output
    assert "example" not in output
    path = tmp_path / "detail.csv"
    export_csv(path, [row])
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["alternative_q"] == ""
    assert saved[0]["q_margin"] == ""
    assert saved[0]["mate_q"] == "1.0"


def test_empty_sets_summary(tmp_path, capsys):
    records, counts, overlap = analyze_sets(FixedScores({}), [], [])
    print_summary(records, counts, overlap)
    assert "N/A(n=0)" in capsys.readouterr().out
    export_csv(tmp_path / "empty.csv", records)
    assert (tmp_path / "empty.csv").read_text().strip() == "dataset,fen,overlap"


def test_cli_reads_real_files_and_checkpoint_without_training(tmp_path, monkeypatch, capsys):
    import csv
    import chess.pgn
    from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
    from chess_rl.training.checkpoint import save_training_checkpoint
    from chess_rl.utils.replay_buffer import ReplayBuffer

    board = capture_mate_board()
    csv_path = tmp_path / "seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fen"])
        writer.writeheader()
        writer.writerow({"fen": board.fen()})
    board.push_uci("h5f7")
    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "DQN"
    game.headers["Black"] = "RandomAgent"
    games = tmp_path / "games"
    games.mkdir()
    (games / "game.pgn").write_text(str(game), encoding="utf-8")
    checkpoint = tmp_path / "agent.pt"
    agent = StateActionDQNAgent(spatial=True)
    save_training_checkpoint(str(checkpoint), agent, ReplayBuffer(10))
    original_bytes = checkpoint.read_bytes()

    def fail(*args, **kwargs):
        pytest.fail("Diagnostic must not train or synchronize weights")
    monkeypatch.setattr(StateActionDQNAgent, "train_step", fail)
    monkeypatch.setattr(StateActionDQNAgent, "update_target", fail)
    output = tmp_path / "output.csv"
    main([str(checkpoint), str(games), "--seed-csv", str(csv_path), str(csv_path),
          "--output-csv", str(output)])
    console = capsys.readouterr().out
    assert "Overlap: 1" in console
    assert "seeded: input=2 duplicates=1 skipped=0 analyzed=1" in console
    assert "games: input=1" in console
    assert board.board_fen() not in console
    assert checkpoint.read_bytes() == original_bytes
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(r["overlap"] == "True" for r in rows)
    with pytest.raises(SystemExit):
        main([str(checkpoint), str(games), "--seed-csv", str(csv_path),
              "--output-csv", str(csv_path)])


from scripts.diagnose_mate_q import expected_alternative_target


class TargetValues(torch.nn.Module):
    def __init__(self, expected_boards, values):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.expected_boards = expected_boards
        self.values = values
        self.calls = 0

    def evaluate_legal_action_maxes(self, states, actions):
        from chess_rl.utils.action_encoder import get_legal_actions
        board = self.expected_boards[self.calls]
        assert not torch.is_grad_enabled()
        assert not self.training
        torch.testing.assert_close(states[0], encode_board(board))
        assert actions == [get_legal_actions(board)]
        value = self.values[self.calls]
        self.calls += 1
        return torch.tensor([value])


@pytest.mark.parametrize("black", [False, True])
def test_expected_target_uniform_replies_and_full_material_change(black):
    # White takes a pawn; some replies recapture the rook.
    board = chess.Board("4k3/8/8/3rp3/8/8/4R3/4K3 w - - 0 1")
    move = chess.Move.from_uci("e2e5")
    if black:
        board = board.mirror()
        move = chess.Move.from_uci("e7e4")
    after = board.copy()
    after.push(move)
    successors, rewards = [], []
    for reply in list(after.legal_moves):
        end = after.copy()
        end.push(reply)
        assert not end.is_game_over()
        successors.append(end)
        # Pawn gain is +.01; recapture loses our rook (-.05).
        rewards.append(0.01 - (0.05 if reply.to_square == move.to_square else 0.0))
    assert min(rewards) == pytest.approx(-0.04)
    values = [i / 8 for i in range(len(successors))]
    net = TargetValues(successors, values)
    before = board.fen(), list(board.move_stack)
    result = expected_alternative_target(board, move, net, 0.9)
    expected = sum(r - 0.0005 + 0.9 * q for r, q in zip(rewards, values)) / len(values)
    assert result["alternative_expected_target"] == pytest.approx(expected)
    assert result["opponent_reply_count"] == len(values)
    assert result["opponent_terminal_fraction"] == 0
    assert net.calls == len(values)
    assert (board.fen(), list(board.move_stack)) == before


@pytest.mark.parametrize("black", [False, True])
def test_expected_target_mixed_mating_and_nonterminal_replies(black):
    board = chess.Board()
    for uci in ("f2f3", "e7e5"):
        board.push_uci(uci)
    move = chess.Move.from_uci("g2g4")
    if black:
        board = board.mirror()
        move = chess.Move.from_uci("g7g5")
    after = board.copy()
    after.push(move)
    ends = []
    for reply in list(after.legal_moves):
        end = after.copy()
        end.push(reply)
        ends.append(end)
    terminal = [end for end in ends if end.is_game_over()]
    live = [end for end in ends if not end.is_game_over()]
    assert len(terminal) == 1 and terminal[0].is_checkmate()
    net = TargetValues(live, [0.5] * len(live))
    result = expected_alternative_target(board, move, net, 0.9)
    assert result["alternative_expected_target"] == pytest.approx(
        (-1 + len(live) * (-0.0005 + 0.9 * 0.5)) / len(ends)
    )
    assert result["opponent_terminal_fraction"] == pytest.approx(1 / len(ends))
    assert net.calls == len(live)


@pytest.mark.parametrize("black", [False, True])
def test_expected_immediate_terminal_has_no_bootstrap(black):
    board = capture_mate_board(black)
    move = chess.Move.from_uci("h4f2" if black else "h5f7")
    net = TargetValues([], [])
    result = expected_alternative_target(board, move, net, 0.99)
    assert result == dict(alternative_expected_target=1.01,
                          opponent_reply_count=0, opponent_terminal_fraction=None)
    assert net.calls == 0


def test_expected_reply_preserves_repetition_history():
    board = chess.Board()
    cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
    for uci in cycle * 3 + cycle[:2]:
        board.push_uci(uci)
    before = board.fen(), list(board.move_stack)
    move = chess.Move.from_uci("f3g1")
    after = board.copy()
    after.push(move)
    live = []
    for reply in list(after.legal_moves):
        end = after.copy()
        end.push(reply)
        if not end.is_game_over():
            live.append(end)
    n = after.legal_moves.count()
    assert len(live) == n - 1
    net = TargetValues(live, [0.0] * len(live))
    result = expected_alternative_target(board, move, net, 0.99)
    assert result["opponent_terminal_fraction"] == pytest.approx(1 / n)
    assert result["alternative_expected_target"] == pytest.approx(-0.0005 * (n - 1) / n)
    assert (board.fen(), list(board.move_stack)) == before


def test_expected_target_does_not_evaluate_policy_or_train(monkeypatch):
    from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
    agent = StateActionDQNAgent(spatial=True)
    def fail(*args, **kwargs):
        pytest.fail("Unexpected policy evaluation or training")
    monkeypatch.setattr(agent, "train_step", fail)
    monkeypatch.setattr(agent, "update_target", fail)
    monkeypatch.setattr(agent.optimizer, "step", fail)
    monkeypatch.setattr(agent.policy_net, "evaluate_action_ids", fail)
    monkeypatch.setattr(agent.policy_net, "evaluate_legal_action_maxes", fail)
    before = [{k: v.clone() for k,v in net.state_dict().items()}
              for net in (agent.policy_net, agent.target_net)]
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    result = expected_alternative_target(board, chess.Move.from_uci("f7f6"), agent.target_net, 0.99)
    assert result["opponent_reply_count"] > 0
    for net, saved in zip((agent.policy_net, agent.target_net), before):
        assert all(torch.equal(v, saved[k]) for k,v in net.state_dict().items())
        assert all(p.grad is None for p in net.parameters())


def test_expected_immediate_draw():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    result = expected_alternative_target(board, chess.Move.from_uci("f7e6"), TargetValues([], []), 0.99)
    assert result == dict(alternative_expected_target=0.0,
                          opponent_reply_count=0, opponent_terminal_fraction=None)


@pytest.mark.parametrize("gamma", [-0.1, 1.1, float("nan")])
def test_expected_target_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError, match="gamma"):
        expected_alternative_target(chess.Board(), chess.Move.from_uci("e2e4"), TargetValues([], []), gamma)
