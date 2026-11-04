from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from chess_rl.utils.replay_buffer import Transition
from scripts import diagnose_mate_replay as diagnostic


@pytest.fixture
def transition():
    return Transition(torch.zeros(18, 8, 8), 1, 1.0,
                      torch.ones(18, 8, 8), True, [])


def match(monkeypatch, examples, replay):
    def load(path):
        assert path == "mates.csv"
        return examples, 1
    monkeypatch.setattr(diagnostic, "load_mate_examples", load)
    return diagnostic.match_mate_replay("mates.csv", SimpleNamespace(buffer=replay))


def test_matches_equal_content_not_identity(monkeypatch, transition):
    other = deepcopy(transition)
    result = match(monkeypatch, [transition], [other])
    assert other is not transition and other.state is not transition.state
    assert result["generated_count"] == result["matched_count"] == 1
    assert result["missing_count"] == 0
    assert result["compatible_replay_count"] == 1
    assert result["compatible_replay_indices"] == [0]
    assert result["example_matches"] == result["replay_matches"] == [[0]]
    assert result["example_duplicate_groups"] == result["replay_duplicate_groups"] == []
    assert result["ambiguous_example_indices"] == result["ambiguous_replay_indices"] == []


@pytest.mark.parametrize("field", [
    "state", "action", "reward", "next_state", "done", "next_legal_actions",
])
def test_difference_in_each_field_prevents_match(monkeypatch, transition, field):
    other = deepcopy(transition)
    if field in ("state", "next_state"):
        getattr(other, field)[0, 0, 0] += 1
    else:
        setattr(other, field, {"action": 2, "reward": 1.01, "done": False,
                              "next_legal_actions": [1]}[field])
    result = match(monkeypatch, [transition], [other])
    assert result["matched_count"] == result["compatible_replay_count"] == 0
    assert result["missing_count"] == 1
    assert result["example_matches"] == result["replay_matches"] == [[]]


def test_duplicates_and_ambiguity_in_both_directions(monkeypatch, transition):
    unmatched = deepcopy(transition)
    unmatched.reward = -1.0
    result = match(monkeypatch,
                   [transition, deepcopy(transition), unmatched],
                   [deepcopy(transition), deepcopy(unmatched), deepcopy(transition),
                    deepcopy(unmatched)])
    assert result["generated_count"] == result["matched_count"] == 3
    assert result["compatible_replay_count"] == 4
    assert result["example_matches"] == [[0, 2], [0, 2], [1, 3]]
    assert result["replay_matches"] == [[0, 1], [2], [0, 1], [2]]
    assert result["example_duplicate_groups"] == [[0, 1]]
    assert result["replay_duplicate_groups"] == [[0, 2], [1, 3]]
    assert result["ambiguous_example_indices"] == [0, 1, 2]
    assert result["ambiguous_replay_indices"] == [0, 2]


def test_unmatched_duplicates_are_not_ambiguous(monkeypatch, transition):
    other = deepcopy(transition)
    other.reward = -1.0
    result = match(monkeypatch, [transition, deepcopy(transition)],
                   [other, deepcopy(other)])
    assert result["missing_count"] == 2
    assert result["example_duplicate_groups"] == result["replay_duplicate_groups"] == [[0, 1]]
    assert result["ambiguous_example_indices"] == result["ambiguous_replay_indices"] == []
    assert result["compatible_replay_indices"] == []


def test_empty_replay(monkeypatch, transition):
    result = match(monkeypatch, [transition], [])
    assert result["missing_count"] == 1
    assert result["replay_matches"] == []


def test_real_csv_loader_deduplicates_positions(tmp_path):
    path = tmp_path / "mates.csv"
    row = "rnbqkbnr/ppppp2p/5p2/6p1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3,d1h5\n"
    path.write_text("fen,mate_moves\n" + row * 2, encoding="utf-8")
    examples, _ = diagnostic.load_mate_examples(path)
    replay = SimpleNamespace(buffer=deepcopy(examples))
    before = deepcopy(replay.buffer)
    result = diagnostic.match_mate_replay(path, replay)
    assert result["unique_csv_positions"] == result["generated_count"] == 1
    assert result["matched_count"] == result["compatible_replay_count"] == 1
    assert result["example_duplicate_groups"] == []
    assert torch.equal(replay.buffer[0].state, before[0].state)
    assert torch.equal(replay.buffer[0].next_state, before[0].next_state)


def test_summary_exclusive_categories_and_one_calculation(monkeypatch, transition):
    rows = [deepcopy(transition) for _ in range(7)]
    labels = ["win", "win", "draw", "loss", "truncated", None, "win"]
    rows[4].done = rows[5].done = False
    tracker = SimpleNamespace(get_final_labels=lambda: {
        id(t): label for t, label in zip(rows, labels) if label is not None})
    monkeypatch.setattr(diagnostic, "match_mate_replay", lambda *args: {
        "compatible_replay_indices": [0, 6]})
    calls = []
    def calculate(*args, **kwargs):
        calls.append((args, kwargs))
        return {"q": torch.tensor([0., 1., 2., 3., 4., 5., 2.]),
                "target": torch.ones(7),
                "td_error": torch.tensor([1., 0., -1., -2., -3., -4., -1.]),
                "abs_td_error": torch.tensor([1., 0., 1., 2., 3., 4., 1.])}
    monkeypatch.setattr(diagnostic, "calculate_replay_learning_signal", calculate)
    result = diagnostic.summarize_replay(SimpleNamespace(gamma=0.99),
                                         SimpleNamespace(buffer=rows), tracker, None)
    assert len(calls) == 1
    assert sum(group["count"] for group in result.values()) == 7
    assert result["compatible_with_seeded_mate"] == {
        "count": 2, "q": 1., "target": 1., "td_error": 0., "abs_td_error": 1.}
    for name, q in (("other_win", 1), ("draw", 2), ("loss", 3),
                    ("truncated", 4), ("non_terminal", 5)):
        assert result[name] == dict(count=1, q=q, target=1, td_error=1-q, abs_td_error=abs(1-q))


@pytest.mark.parametrize("label,done", [("draw", True), ("win", False)])
def test_summary_rejects_incompatible_labels(monkeypatch, transition, label, done):
    transition.done = done
    monkeypatch.setattr(diagnostic, "match_mate_replay", lambda *args: {
        "compatible_replay_indices": [0]})
    tracker = SimpleNamespace(get_final_labels=lambda: {id(transition): label})
    with pytest.raises(ValueError):
        diagnostic.summarize_replay(SimpleNamespace(gamma=.99),
                                    SimpleNamespace(buffer=[transition]), tracker, None)


def evaluation_csv(tmp_path, **changes):
    import csv
    row = dict(fen="example", dataset="seeded", mate_selected="True", overlap="False",
               alternative_category="no_alternative", mate_q="0.8", alternative_q="",
               q_margin="", seeded_target="1", seeded_error="-0.2",
               seeded_absolute_error="0.2", game_mate_reward="1")
    row.update(changes)
    path = tmp_path / "detail.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path


@pytest.mark.parametrize("changes", [
    {"dataset": "unknown"}, {"mate_selected": "yes"}, {"overlap": "0"},
    {"mate_q": "nan"}, {"mate_q": "bad"}, {"mate_q": ""},
    {"alternative_category": "invalid"}, {"alternative_q": "1"},
    {"alternative_category": "non_terminal"},
])
def test_evaluation_validation(tmp_path, changes):
    with pytest.raises(ValueError):
        diagnostic.read_evaluation_detail(evaluation_csv(tmp_path, **changes))


def test_evaluation_empty_missing_columns_and_cached_values(tmp_path):
    path = evaluation_csv(tmp_path)
    rows = diagnostic.read_evaluation_detail(path)
    assert rows[0]["mate_q"] == "0.8"
    assert rows[0]["alternative_q"] == ""
    content = path.read_text()
    path.write_text(content + content.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        diagnostic.read_evaluation_detail(path)
    path.write_text(content.splitlines()[0] + "\n")
    assert diagnostic.read_evaluation_detail(path) == []
    path.write_text("fen,dataset\n")
    with pytest.raises(ValueError, match="columns"):
        diagnostic.read_evaluation_detail(path)


def test_readonly_cli_and_empty_categories(tmp_path, monkeypatch, capsys, transition):
    import random
    import numpy as np
    from chess_rl.utils.replay_buffer import ReplayBuffer

    with torch.device("meta"):
        network = diagnostic.SpatialStateActionDQN()
    weights = {key: torch.zeros(value.shape, dtype=value.dtype)
               for key, value in network.state_dict().items()}
    checkpoint = {
        "agent": {"policy_net": weights, "target_net": deepcopy(weights)},
        "replay_buffer": {"capacity": 2, "transitions": [transition], "priorities": [0.7]},
        "metadata": {"architecture": "spatial", "terminal_tracker": {"terminal_labels": ["win"]}},
    }
    monkeypatch.setattr(diagnostic.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(diagnostic, "match_mate_replay", lambda *args: {"compatible_replay_indices": [0]})
    original = diagnostic.calculate_replay_learning_signal
    calls = []
    def calculate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(diagnostic, "calculate_replay_learning_signal", calculate)
    before = deepcopy(checkpoint)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    diagnostic.main(["unused.pt", "--mate-csv", "unused.csv", "--existing-detail",
                     str(evaluation_csv(tmp_path)), "--gamma", "0.99"])
    output = capsys.readouterr().out
    assert calls == [1]
    assert "compatible_with_seeded_mate: n=1 q=0.000000 target=1.000000 td_error=1.000000 abs_td_error=1.000000" in output
    assert "other_win: n=0 q=N/A" in output
    assert "Qmate=0.800000(n=1)" in output
    assert "Qalt=N/A(n=0)" in output
    assert "games: input=0" in output
    assert "Independence from training/replay has not been verified" in output
    assert checkpoint["replay_buffer"]["priorities"] == before["replay_buffer"]["priorities"]
    for net in ("policy_net", "target_net"):
        for key, value in checkpoint["agent"][net].items():
            assert torch.equal(value, before["agent"][net][key])
    assert torch.equal(transition.state, before["replay_buffer"]["transitions"][0].state)
    assert random.getstate() == rng[0]
    after = np.random.get_state()
    assert after[0] == rng[1][0] and np.array_equal(after[1], rng[1][1]) and after[2:] == rng[1][2:]
    assert torch.equal(torch.get_rng_state(), rng[2])
    replay = ReplayBuffer(capacity=1)
    tracker = diagnostic.TerminalTransitionTracker(replay)
    monkeypatch.setattr(diagnostic, "match_mate_replay", lambda *args: {"compatible_replay_indices": []})
    with pytest.raises(ValueError, match="empty replay"):
        diagnostic.summarize_replay(SimpleNamespace(gamma=.99), replay, tracker, None)
    with pytest.raises(ValueError, match="gamma"):
        diagnostic.summarize_replay(SimpleNamespace(gamma=.9), replay, tracker, None)
