from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.models.dqn_cnn import DQNCNN
import chess
from chess_rl.utils.action_encoder import decode_legal_action
from chess_rl.utils.action_selection import select_greedy_action
from chess_rl.utils.board_encoder import encode_board

def create_frozen_opponent(
    agent: DQNAgent,
) -> DQNCNN:
    """
    Create an independent frozen copy of the agent's current policy.

    The returned network can be used as a stable self-play opponent.
    It does not share parameters with the learning policy and its
    parameters do not require gradients.
    """
    opponent = DQNCNN()

    opponent.load_state_dict(
        agent.policy_net.state_dict()
    )

    opponent.eval()

    for parameter in opponent.parameters():
        parameter.requires_grad_(False)

    return opponent

def select_frozen_opponent_move(
    opponent: DQNCNN,
    board: chess.Board,
) -> chess.Move:
    """
    Select the frozen opponent's best legal move greedily.
    """
    legal_moves = list(board.legal_moves)

    state = encode_board(board)

    action = select_greedy_action(
        network=opponent,
        state=state,
        legal_moves=legal_moves,
    )

    return decode_legal_action(
        action=action,
        legal_moves=legal_moves,
    )