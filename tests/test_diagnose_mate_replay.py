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
