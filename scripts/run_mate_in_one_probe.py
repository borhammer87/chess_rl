
import chess
from chess_rl.utils.action_encoder import decode_legal_action, encode_move
from chess_rl.utils.action_selection import evaluate_state_action_greedy_choice
from chess_rl.utils.board_encoder import encode_board
import torch
from chess_rl.utils.replay_buffer import ReplayBuffer

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

def build_mate_transitions(board: chess.Board):
    """Build real terminal transitions for immediate checkmates."""
    from chess_rl.utils.replay_buffer import Transition

    transitions = []
    state = encode_board(board)

    for move in find_mate_in_one_moves(board):
        next_board = board.copy()
        next_board.push(move)

        assert next_board.is_checkmate()

        transitions.append(
            Transition(
                state=state.clone(),
                action=encode_move(move),
                reward=1.0,
                next_state=encode_board(next_board),
                done=True,
                next_legal_actions=[],
            )
        )

    return transitions

def build_mate_replay(positions: list[chess.Board]) -> ReplayBuffer:
    """Create a replay buffer containing genuine mate transitions."""
    transitions = [
        transition
        for board in positions
        for transition in build_mate_transitions(board)
    ]

    if not transitions:
        raise ValueError("No mate transitions available.")

    replay = ReplayBuffer(capacity=len(transitions))

    for transition in transitions:
        replay.push(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            next_state=transition.next_state,
            done=transition.done,
            next_legal_actions=transition.next_legal_actions,
        )

    return replay

def train_mate_batch(
    agent: StateActionDQNAgent,
    positions: list[chess.Board],
) -> float:
    """Train on real terminal mate-in-one transitions."""
    if not positions:
        raise ValueError("Training positions cannot be empty.")

    transitions = [
        transition
        for board in positions
        for transition in build_mate_transitions(board)
    ]

    if not transitions:
        raise ValueError("No mate-in-one transitions found.")

    return agent.train_step(transitions)

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


def run_mate_in_one_probe(network, positions=None):
    """Evaluate all mate-in-one positions and report accuracy."""
    correct = 0
    if positions is None:
        positions = MATE_POSITIONS

    total = len(positions)

    for name, fen in positions.items():
        board = chess.Board(fen)
        (
            move,
            is_mate,
            best_q,
            q_gap,
            mate_q,
            non_mate_q,
            mate_advantage,
        ) = evaluate_mate_position(board, network)

        correct += is_mate
        status = "PASS" if is_mate else "FAIL"
        gap_text = "N/A" if q_gap is None else f"{q_gap:.4f}"

        def format_q(value):
            return "N/A" if value is None else f"{value:.4f}"

        print(
            f"{name}: {status} | move={move.uci()} "
            f"| Q={best_q:.4f} | Q gap={gap_text} "
            f"| mate Q={format_q(mate_q)} "
            f"| best non-mate Q={format_q(non_mate_q)} "
            f"| mate advantage={format_q(mate_advantage)}"
        )

    print(f"Mate-in-one accuracy: {correct}/{total} ({correct / total:.1%})")
    return correct, total

def run_mate_learning_probe(
    seed: int = 12345,
    updates: int = 50,
):
    """Test whether the spatial network learns terminal mate moves."""
    if updates < 1:
        raise ValueError("Updates must be positive.")

    torch.manual_seed(seed)
    agent = StateActionDQNAgent(spatial=True, device="cpu")

    held_out_name = "fools_mate_white"
    training_fens = {
        name: fen
        for name, fen in MATE_POSITIONS.items()
        if name != held_out_name
    }
    held_out_fens = {
        held_out_name: MATE_POSITIONS[held_out_name]
    }

    positions = [
        chess.Board(fen)
        for fen in training_fens.values()
    ]

    print(f"Mate learning probe - seed: {seed}")
    print("\nTraining positions - before:")
    run_mate_in_one_probe(agent.policy_net, training_fens)

    print("\nHeld-out position - before:")
    initial = run_mate_in_one_probe(agent.policy_net, held_out_fens)

    losses = [
        train_mate_batch(agent, positions)
        for _ in range(updates)
    ]

    print(f"\nTraining updates: {updates}")
    print(f"First loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")

    print("\nTraining positions - after:")
    run_mate_in_one_probe(agent.policy_net, training_fens)

    print("\nHeld-out position - after:")
    final = run_mate_in_one_probe(agent.policy_net, held_out_fens)

    return initial, final

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
