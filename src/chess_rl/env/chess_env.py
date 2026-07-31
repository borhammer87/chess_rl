import chess


class ChessEnv:
    """
    Minimal chess environment based on python-chess.

    Responsibilities:
    - Maintain the current board.
    - Reset the game.
    - Expose legal moves.
    - Apply legal moves.
    - Return terminal rewards.
    """

    def __init__(self) -> None:
        self.board = chess.Board()
        self.done = False

    def reset(self) -> chess.Board:
        """
        Reset the environment to the initial chess position.
        """
        self.board.reset()
        self.done = False

        return self.get_state()

    def get_state(self) -> chess.Board:
        """
        Return a copy of the current board state.

        Returning a copy prevents external code from modifying
        the environment's internal board accidentally.
        """
        return self.board.copy(stack=False)

    def legal_moves(self) -> list[chess.Move]:
        """
        Return all legal moves in the current position.
        """
        return list(self.board.legal_moves)

    def step(
        self,
        move: chess.Move,
    ) -> tuple[chess.Board, float, bool, dict]:
        """
        Apply one legal chess move.

        Args:
            move: Move to apply.

        Returns:
            next_state: Copy of the board after the move.
            reward: Terminal reward from White's perspective.
            done: Whether the game has ended.
            info: Additional environment information.
        """
        if self.done:
            raise RuntimeError("Cannot apply a move: the game has already ended.")

        if not isinstance(move, chess.Move):
            raise TypeError("move must be an instance of chess.Move.")

        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move.uci()}")

        self.board.push(move)
        self.done = self.board.is_game_over()

        reward = self._calculate_reward()

        info = {
            "result": self.board.result() if self.done else None,
            "termination": (
                self.board.outcome().termination.name
                if self.done and self.board.outcome() is not None
                else None
            ),
        }

        return self.get_state(), reward, self.done, info

    def _calculate_reward(self) -> float:
        """
        Return the terminal reward from White's perspective.

        White win: +1
        Black win: -1
        Draw or unfinished game: 0
        """
        if not self.done:
            return 0.0

        result = self.board.result()

        if result == "1-0":
            return 1.0

        if result == "0-1":
            return -1.0

        return 0.0