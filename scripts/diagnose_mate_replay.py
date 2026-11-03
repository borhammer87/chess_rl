"""Identify replay content compatible with generated mate examples, not provenance."""
from collections import defaultdict

from scripts.inject_mate_replay import load_mate_examples


def _transition_key(transition):
    """Use complete encoded content, including tensor shape and dtype."""
    def tensor_key(tensor):
        return (tuple(tensor.shape), str(tensor.dtype),
                tensor.detach().cpu().contiguous().numpy().tobytes())

    return (
        tensor_key(transition.state), transition.action, transition.reward,
        tensor_key(transition.next_state), transition.done,
        tuple(transition.next_legal_actions),
    )


def match_mate_replay(csv_path, replay_buffer):
    """Return all content matches and duplicate groups, using zero-based indices.

    CSV indices refer to transitions returned by load_mate_examples, which
    already deduplicates CSV positions. Multiple matches are retained rather
    than assigned an arbitrary one-to-one correspondence. Compatibility does
    not establish whether a replay transition was originally preloaded.
    """
    examples, unique_positions = load_mate_examples(csv_path)
    example_groups = defaultdict(list)
    replay_groups = defaultdict(list)
    for index, transition in enumerate(examples):
        example_groups[_transition_key(transition)].append(index)
    for index, transition in enumerate(replay_buffer.buffer):
        replay_groups[_transition_key(transition)].append(index)

    example_matches = [[] for _ in examples]
    replay_matches = [[] for _ in replay_buffer.buffer]
    for key, indices in example_groups.items():
        matches = replay_groups.get(key, [])
        for index in indices:
            example_matches[index] = list(matches)
        for index in matches:
            replay_matches[index] = list(indices)

    matched = sum(bool(matches) for matches in example_matches)
    return {
        "unique_csv_positions": unique_positions,
        "generated_count": len(examples),
        "matched_count": matched,
        "missing_count": len(examples) - matched,
        "compatible_replay_indices": [
            i for i, matches in enumerate(replay_matches) if matches
        ],
        "compatible_replay_count": sum(bool(matches) for matches in replay_matches),
        "example_matches": example_matches,
        "replay_matches": replay_matches,
        "example_duplicate_groups": [
            indices for indices in example_groups.values() if len(indices) > 1
        ],
        "replay_duplicate_groups": [
            indices for indices in replay_groups.values() if len(indices) > 1
        ],
        "ambiguous_example_indices": [
            i for i, matches in enumerate(example_matches) if len(matches) > 1
        ],
        "ambiguous_replay_indices": [
            i for i, matches in enumerate(replay_matches) if len(matches) > 1
        ],
    }
