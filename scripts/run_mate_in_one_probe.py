
import chess
from chess_rl.utils.action_encoder import decode_legal_action
from chess_rl.utils.action_selection import evaluate_state_action_greedy_choice
from chess_rl.utils.board_encoder import encode_board
import torch

from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent

def find_mate_in_one_moves(board: chess.Board) -> list[chess.Move]:
    """Return all legal moves that immediately checkmate."""
    mates = []

    for move in list(board.legal_moves):
        board.push(move)
        if board.is_checkmate():
            mates.append(move)
        board.pop()

    return mates


MATE_POSITIONS = {
    "fools_mate_black": (
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/"
        "PPPPP2P/RNBQKBNR b KQkq g3 0 2"
    ),
    "fools_mate_white": (
        "rnbqkbnr/ppppp2p/5p2/6p1/4P3/8/"
        "PPPP1PPP/RNBQKBNR w KQkq g6 0 3"
    ),
    "queen_rook_mate_white": (
        "6k1/5ppp/8/8/8/8/6PP/4Q1KR w - - 0 1"
    ),
    "queen_rook_mate_black": (
        "4q1kr/6pp/8/8/8/8/5PPP/6K1 b - - 0 1"
    ),
}



def evaluate_mate_position(board, network):
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

    return chosen_move, chosen_move in mate_moves, best_q, q_gap


def run_mate_in_one_probe(network):
    """Evaluate all mate-in-one positions and report accuracy."""
    correct = 0
    total = len(MATE_POSITIONS)

    for name, fen in MATE_POSITIONS.items():
        board = chess.Board(fen)
        move, is_mate, best_q, q_gap = evaluate_mate_position(
            board, network
        )

        correct += is_mate
        status = "PASS" if is_mate else "FAIL"
        gap_text = "N/A" if q_gap is None else f"{q_gap:.4f}"

        print(
            f"{name}: {status} | move={move.uci()} "
            f"| Q={best_q:.4f} | Q gap={gap_text}"
        )

    print(f"Mate-in-one accuracy: {correct}/{total} ({correct / total:.1%})")
    return correct, total


def main():
    """Evaluate an untrained spatial State-Action DQN."""
    seed = 12345
    torch.manual_seed(seed)

    agent = StateActionDQNAgent(
        spatial=True,
        device="cpu",
    )
    agent.policy_net.eval()

    print(f"Mate-in-one probe - spatial - untrained - seed: {seed}")
    run_mate_in_one_probe(agent.policy_net)


if __name__ == "__main__":
    main()
