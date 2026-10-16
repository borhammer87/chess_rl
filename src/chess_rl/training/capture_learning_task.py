
import chess
import random
from chess_rl.training.episodes import material_reward
import torch
from chess_rl.utils.replay_buffer import Transition
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.board_encoder import encode_board
from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent

def best_material_capture(
    board: chess.Board,
    first_move: chess.Move,
    second_move: chess.Move,
) -> chess.Move:
    """Return the capture with the greater immediate material reward."""
    if first_move == second_move:
        raise ValueError("Candidate moves must be different.")

    rewards = []

    for move in (first_move, second_move):
        if move not in board.legal_moves or not board.is_capture(move):
            raise ValueError("Both moves must be legal captures.")

        next_board = board.copy()
        next_board.push(move)

        rewards.append(
            material_reward(board, next_board, board.turn)
        )

    if rewards[0] == rewards[1]:
        raise ValueError("Captures must have different material rewards.")

    return first_move if rewards[0] > rewards[1] else second_move


def make_capture_positions() -> list[
    tuple[chess.Board, chess.Move, chess.Move]
]:
    """Build a small deterministic set of capture comparisons."""
    positions = []

    for knight_file in (2, 3, 4, 5):
        knight_square = chess.square(knight_file, 2)
        left_target = chess.square(knight_file - 1, 4)
        right_target = chess.square(knight_file + 1, 4)

        for rook_on_left in (True, False):
            board = chess.Board(None)
            board.turn = chess.WHITE
            board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(
                knight_square, chess.Piece(chess.KNIGHT, chess.WHITE)
            )
            board.set_piece_at(
                left_target,
                chess.Piece(
                    chess.ROOK if rook_on_left else chess.PAWN,
                    chess.BLACK,
                ),
            )
            board.set_piece_at(
                right_target,
                chess.Piece(
                    chess.PAWN if rook_on_left else chess.ROOK,
                    chess.BLACK,
                ),
            )

            positions.append((
                board,
                chess.Move(knight_square, left_target),
                chess.Move(knight_square, right_target),
            ))

    return positions


def generate_capture_positions(
    count: int,
    seed: int,
) -> list[tuple[chess.Board, chess.Move, chess.Move]]:
    """Generate reproducible, distinct capture comparisons."""
    if count < 0:
        raise ValueError("Count cannot be negative.")
    if count > 12000:
        raise ValueError(
            "Requested count exceeds the generator's "
            "theoretical maximum of 12000 distinct positions."
        )
    rng = random.Random(seed)
    positions = []
    seen = set()
    attempts = 0

    while len(positions) < count and attempts < max(1000, count * 100):
        attempts += 1
        color = rng.choice((chess.WHITE, chess.BLACK))
        file = rng.randint(1, 6)
        rank = rng.randint(1, 5) if color else rng.randint(2, 6)
        target_rank = rank + 2 if color else rank - 2

        origin = chess.square(file, rank)
        left = chess.square(file - 1, target_rank)
        right = chess.square(file + 1, target_rank)

        board = chess.Board(None)
        board.turn = color
        white_king = rng.choice((
            chess.A1, chess.B1, chess.C1,
            chess.A2, chess.B2, chess.C2,
        ))

        black_king = rng.choice((
            chess.F7, chess.G7, chess.H7,
            chess.F8, chess.G8, chess.H8,
        ))

        board.set_piece_at(
            white_king, chess.Piece(chess.KING, chess.WHITE)
        )
        board.set_piece_at(
            black_king, chess.Piece(chess.KING, chess.BLACK)
        )
        board.set_piece_at(origin, chess.Piece(chess.KNIGHT, color))

        targets = [chess.ROOK, chess.PAWN]
        rng.shuffle(targets)
        for square, piece_type in zip((left, right), targets):
            board.set_piece_at(
                square, chess.Piece(piece_type, not color)
            )

        first = chess.Move(origin, left)
        second = chess.Move(origin, right)
        if rng.choice((True, False)):
            first, second = second, first

        if not board.is_valid():
            continue
        if first not in board.legal_moves or second not in board.legal_moves:
            continue

        key = board.fen()
        if key in seen:
            continue

        seen.add(key)
        positions.append((board, first, second))

    if len(positions) != count:
        raise ValueError("Could not generate enough distinct positions.")

    return positions


def split_capture_positions(
    train_count: int,
    evaluation_count: int,
    seed: int,
) -> tuple[
    list[tuple[chess.Board, chess.Move, chess.Move]],
    list[tuple[chess.Board, chess.Move, chess.Move]],
]:
    """Create disjoint, balanced training and evaluation sets."""
    if train_count < 0 or evaluation_count < 0:
        raise ValueError("Dataset sizes cannot be negative.")

    positions = generate_capture_positions(
        train_count + evaluation_count,
        seed,
    )

    rng = random.Random(seed)
    rng.shuffle(positions)

    train = positions[:train_count]
    evaluation = positions[train_count:]

    def balance_candidate_order(
        examples: list[tuple[chess.Board, chess.Move, chess.Move]],
    ) -> list[tuple[chess.Board, chess.Move, chess.Move]]:
        balanced = []

        for index, (board, first, second) in enumerate(examples):
            best = best_material_capture(board, first, second)
            should_be_first = index % 2 == 0

            if (best == first) != should_be_first:
                first, second = second, first

            balanced.append((board, first, second))

        return balanced

    return (
        balance_candidate_order(train),
        balance_candidate_order(evaluation),
    )


def evaluate_capture_accuracy(
    model: torch.nn.Module,
    positions: list[tuple[chess.Board, chess.Move, chess.Move]],
) -> float:
    """Measure correct capture preferences without training."""
    if not positions:
        raise ValueError("Evaluation positions cannot be empty.")

    correct = 0
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for board, first, second in positions:
                state = encode_board(board)
                action_ids = [encode_move(first), encode_move(second)]
                scores = model.evaluate_action_ids(state, action_ids)

                predicted = first if scores[0] > scores[1] else second
                expected = best_material_capture(board, first, second)

                correct += predicted == expected
    finally:
        model.train(was_training)

    return correct / len(positions)


def build_capture_transitions(
    positions: list[tuple[chess.Board, chess.Move, chess.Move]],
) -> list[Transition]:
    """Build terminal one-step experiences for both candidate captures."""
    transitions = []

    for board, first, second in positions:
        best_material_capture(board, first, second)
        state = encode_board(board)

        for move in (first, second):
            next_board = board.copy()
            next_board.push(move)

            transitions.append(
                Transition(
                    state=state.clone(),
                    action=encode_move(move),
                    reward=material_reward(
                        board, next_board, board.turn
                    ),
                    next_state=encode_board(next_board),
                    done=True,
                    next_legal_actions=[],
                )
            )

    return transitions

def train_capture_batch(
    agent: StateActionDQNAgent,
    positions: list[tuple[chess.Board, chess.Move, chess.Move]],
) -> float:
    """Perform one training update using both captures per position."""
    if not positions:
        raise ValueError("Training positions cannot be empty.")

    transitions = build_capture_transitions(positions)
    return agent.train_step(transitions)


def capture_q_gap_summary(
    model: torch.nn.Module,
    positions: list[tuple[chess.Board, chess.Move, chess.Move]],
) -> tuple[float, float]:
    """Return mean correct-minus-incorrect Q gap and error rate."""
    if not positions:
        raise ValueError("Positions cannot be empty.")

    gaps = []
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for board, first, second in positions:
                scores = model.evaluate_action_ids(
                    encode_board(board),
                    [encode_move(first), encode_move(second)],
                )
                best = best_material_capture(board, first, second)
                gap = (
                    scores[0] - scores[1]
                    if best == first
                    else scores[1] - scores[0]
                )
                gaps.append(gap.item())
    finally:
        model.train(was_training)

    return (
        sum(gaps) / len(gaps),
        sum(gap <= 0 for gap in gaps) / len(gaps),
    )


def captured_piece_features(
    board: chess.Board,
    move: chess.Move,
) -> torch.Tensor:
    """One-hot encoding of the captured piece type."""
    if move not in board.legal_moves or not board.is_capture(move):
        raise ValueError("Expected a legal capture.")

    piece = board.piece_at(move.to_square)
    if piece is None:
        raise ValueError("Expected a piece on the destination square.")

    features = torch.zeros(6, dtype=torch.float32)
    features[piece.piece_type - 1] = 1.0
    return features
