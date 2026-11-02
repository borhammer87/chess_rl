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
    source = board.copy()
    source.pop()
    row = analyze_position(source, agent.policy_net)
    detail = tmp_path / "existing.csv"
    export_csv(detail, [dict(row, dataset=label, overlap=True) for label in ("seeded", "games")])
    from chess_rl.models.spatial_state_action_dqn import SpatialStateActionDQN
    monkeypatch.setattr(SpatialStateActionDQN, "evaluate_action_ids",
                        lambda *args: pytest.fail("Must reuse existing Q values"))

    def fail(*args, **kwargs):
        pytest.fail("Diagnostic must not train or synchronize weights")
    monkeypatch.setattr(StateActionDQNAgent, "train_step", fail)
    monkeypatch.setattr(StateActionDQNAgent, "update_target", fail)
    output = tmp_path / "output.csv"
    main([str(checkpoint), str(games), "--seed-csv", str(csv_path), str(csv_path),
          "--existing-detail", str(detail), "--gamma", "0.99", "--output-csv", str(output)])
    console = capsys.readouterr().out
    assert "Overlap: 1" in console
    assert "seeded: sampled=1" in console
    assert "E[y]=" in console
    assert "artificial truncation excluded" in console
    assert "games: sampled=1" in console
    assert board.board_fen() not in console
    assert checkpoint.read_bytes() == original_bytes
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(r["overlap"] == "True" for r in rows)
    with pytest.raises(SystemExit):
        main([str(checkpoint), str(games), "--seed-csv", str(csv_path),
              "--existing-detail", str(detail), "--gamma", "0.99", "--output-csv", str(csv_path)])


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


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_expected_target_restores_mode_even_on_error(training, raises, monkeypatch):
    net = TargetValues([], [])
    net.train(training)
    def evaluate(*args):
        assert not net.training
        if raises:
            raise RuntimeError("inference failed")
        return torch.tensor([0.5])
    monkeypatch.setattr(net, "evaluate_legal_action_maxes", evaluate)
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    if raises:
        with pytest.raises(RuntimeError, match="inference failed"):
            expected_alternative_target(board, chess.Move.from_uci("f7f6"), net, 0.99)
    else:
        expected_alternative_target(board, chess.Move.from_uci("f7f6"), net, 0.99)
    assert net.training is training


from scripts.diagnose_mate_q import select_positions, enrich_existing_detail
from scripts.diagnose_mate_generalization import position_key


def test_hash_sample_is_deterministic_nested_and_deduplicated():
    boards = []
    board = chess.Board()
    for _ in range(35):
        boards.append(board.copy())
        board.push(next(iter(board.legal_moves)))
    unique_count = len({position_key(b) for b in boards})
    selected = select_positions(boards)
    assert len(selected) == min(25, unique_count)
    assert [position_key(b) for b in selected] == [
        position_key(b) for b in select_positions(list(reversed(boards)) + boards)
    ]
    assert select_positions(boards, 5) == select_positions(boards, 10)[:5]
    assert len(select_positions(boards, None)) == unique_count
    assert select_positions([]) == []
    duplicate = boards[0].copy()
    duplicate.fullmove_number += 50
    assert select_positions([boards[0], duplicate], None)[0] is boards[0]
    with pytest.raises(ValueError, match="sample_size"):
        select_positions(boards, 0)


def write_existing_detail(path, board, alternative="f7f6", category="non_terminal"):
    import csv
    row = dict(fen=board.fen(), dataset="seeded", alternative_move=alternative,
               alternative_category=category, alternative_q="0.80000000000000004" if alternative else "",
               mate_q="0.7", q_margin="-0.1", overlap="False")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(dict(row, dataset="games"))
    return row


def test_existing_detail_reuses_q_without_policy_and_preserves_inputs(tmp_path, monkeypatch):
    import scripts.diagnose_mate_q as diagnostic
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    path = tmp_path / "detail.csv"
    original = write_existing_detail(path, board)
    before = path.read_bytes(), board.fen(), list(board.move_stack)
    def fail(*args, **kwargs):
        pytest.fail("Must not recompute policy Q values")
    monkeypatch.setattr(diagnostic, "evaluate_mate_actions", fail)
    monkeypatch.setattr(diagnostic, "analyze_position", fail)
    net = TargetValues([], [])
    monkeypatch.setattr(net, "evaluate_legal_action_maxes", lambda *args: torch.tensor([0.5]))
    records = enrich_existing_detail(path, [board, board.copy()], [board], net, 0.9)
    assert len(records) == 2
    for record in records:
        assert record["alternative_expected_target"] == pytest.approx(-0.0005 + 0.9 * 0.5)
        assert record["alternative_q_minus_target"] == pytest.approx(0.3505)
        for key, value in original.items():
            if key != "dataset":
                assert record[key] == value
    assert (path.read_bytes(), board.fen(), list(board.move_stack)) == before


def test_existing_detail_retains_source_history(tmp_path, monkeypatch):
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    path = tmp_path / "detail.csv"
    write_existing_detail(path, board, "g1f3")
    def expected(source, move, target_net, gamma):
        assert source.move_stack == board.move_stack
        return dict(alternative_expected_target=0.2, opponent_reply_count=20,
                    opponent_terminal_fraction=0.0)
    monkeypatch.setattr("scripts.diagnose_mate_q.expected_alternative_target", expected)
    assert len(enrich_existing_detail(path, [board], [board], None, 0.99)) == 2


@pytest.mark.parametrize("alternative,category", [("", "no_alternative"), ("f7e6", "terminal_draw")])
def test_existing_detail_keeps_terminal_and_absent_alternatives(tmp_path, alternative, category):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    path = tmp_path / "detail.csv"
    write_existing_detail(path, board, alternative, category)
    records = enrich_existing_detail(path, [board], [board], TargetValues([], []), 0.99)
    for record in records:
        assert record["opponent_terminal_fraction"] is None
        assert record["opponent_reply_count"] == (0 if alternative else None)
        assert record["alternative_expected_target"] == (0.0 if alternative else None)


def test_existing_detail_rejects_mismatched_sources_and_illegal_moves(tmp_path):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    path = tmp_path / "detail.csv"
    write_existing_detail(path, board)
    with pytest.raises(ValueError, match="missing"):
        enrich_existing_detail(path, [], [board], None, 0.99)
    changed = board.copy()
    changed.fullmove_number += 1
    with pytest.raises(ValueError, match="FEN"):
        enrich_existing_detail(path, [changed], [board], None, 0.99)
    write_existing_detail(path, board, "a1a2")
    with pytest.raises(ValueError, match="Illegal"):
        enrich_existing_detail(path, [board], [board], None, 0.99)



def test_existing_detail_limits_each_dataset_and_expands_nested(tmp_path, monkeypatch):
    import csv
    boards = []
    for opening in list(chess.Board().legal_moves):
        board = chess.Board()
        board.push(opening)
        for reply in list(board.legal_moves)[:2]:
            end = board.copy()
            end.push(reply)
            boards.append(end)
    assert len({position_key(b) for b in boards}) == 40
    path = tmp_path / "detail.csv"
    rows = [dict(dataset=label, fen=b.fen(), alternative_move=next(iter(b.legal_moves)).uci(),
                 alternative_q="0.8", alternative_category="non_terminal")
            for label in ("seeded", "games") for b in boards]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr("scripts.diagnose_mate_q.expected_alternative_target",
                        lambda *args: dict(alternative_expected_target=0.5,
                                           opponent_reply_count=1, opponent_terminal_fraction=0.0))
    small = enrich_existing_detail(path, boards, boards, None, 0.99)
    large = enrich_existing_detail(path, boards, boards, None, 0.99, sample_size=30)
    assert len(small) == 50 and len(large) == 60
    for label in ("seeded", "games"):
        first = [r["fen"] for r in small if r["dataset"] == label]
        expanded = [r["fen"] for r in large if r["dataset"] == label]
        assert len(first) == 25
        assert first == expanded[:25]


@pytest.mark.parametrize("extra", [[], ["--gamma", "0.99"], ["--existing-detail", "detail.csv"]])
def test_bellman_cli_requires_gamma_and_existing_detail(extra):
    with pytest.raises(SystemExit) as error:
        main(["checkpoint.pt", "games", "--seed-csv", "seed.csv", *extra])
    assert error.value.code == 2


def test_bellman_summary_handles_csv_strings_missing_values_and_signs(capsys):
    base = dict(dataset="seeded", fen="not printed", mate_selected="False",
                mate_q="0.8", alternative_q="0.9", q_margin="-0.1",
                seeded_error="-0.2", seeded_absolute_error="0.2", game_mate_reward="1.0",
                alternative_category="non_terminal", alternative_expected_target=0.7,
                alternative_q_minus_target=0.2, opponent_terminal_fraction=0.5)
    absent = dict(base, alternative_category="no_alternative", alternative_q="", q_margin="",
                  alternative_expected_target=None, alternative_q_minus_target=None,
                  opponent_terminal_fraction=None)
    print_summary([base, absent], {"seeded": {"sampled": 2}}, 0)
    text = capsys.readouterr().out
    assert "E[y]=0.700000(n=1)" in text
    assert "Qalt-E[y]=0.200000(n=1)" in text
    assert "mates=0" in text
    assert "no_alternative: n=1" in text
    assert "E[y]=N/A(n=0)" in text
    assert "not printed" not in text



def test_existing_fen_with_omitted_ep_keeps_original_board_and_q(tmp_path, monkeypatch):
    import chess.pgn
    import scripts.diagnose_mate_q as diagnostic
    # Scholar's mate remains available after a double pawn move without legal EP.
    board = capture_mate_board()
    board.pop()
    board.push_uci("a7a5")
    assert find_mate_in_one_moves(board)
    assert board.ep_square == chess.A6 and not board.has_legal_en_passant()
    assert chess.Board(board.fen()).ep_square is None
    game = chess.pgn.Game.from_board(board)
    board.push_uci("h5f7")
    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "DQN"
    games = tmp_path / "games"
    games.mkdir()
    (games / "game.pgn").write_text(str(game), encoding="utf-8")
    originals = list(diagnostic.game_positions(games))
    source = originals[-1]
    assert source.ep_square == chess.A6
    path = tmp_path / "detail.csv"
    row = write_existing_detail(path, source, "h5h4")
    later = source.copy()
    later.fullmove_number += 10
    def fail(*args, **kwargs):
        pytest.fail("Must not evaluate existing Q values")
    monkeypatch.setattr(diagnostic, "evaluate_mate_actions", fail)
    def expected(actual, move, net, gamma):
        assert actual is source
        assert actual.ep_square == chess.A6
        assert actual.move_stack == source.move_stack
        return dict(alternative_expected_target=0.3, opponent_reply_count=1,
                    opponent_terminal_fraction=0.0)
    monkeypatch.setattr(diagnostic, "expected_alternative_target", expected)
    records = enrich_existing_detail(path, [source, later], [source, later], None, 0.99)
    for record in records:
        assert record["mate_q"] == row["mate_q"]
        assert record["alternative_q"] == row["alternative_q"]
        assert record["alternative_q_minus_target"] == pytest.approx(0.5)


def test_existing_fen_rejects_distinct_original_keys_with_same_serialization(tmp_path):
    board = chess.Board()
    board.push_uci("e2e4")
    without_ep = chess.Board(board.fen())
    assert board.fen() == without_ep.fen()
    assert position_key(board) != position_key(without_ep)
    path = tmp_path / "detail.csv"
    write_existing_detail(path, board, "e7e5")
    with pytest.raises(ValueError, match="Ambiguous detail FEN"):
        enrich_existing_detail(path, [board, without_ep], [board], None, 0.99)
