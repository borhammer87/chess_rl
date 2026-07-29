import chess


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        self.done = False

    def reset(self):
        self.board = chess.Board()
        self.done = False
        return self.get_state()

    def get_state(self):
        """
        Devuelve el estado actual del tablero.
        Por ahora devolvemos el objeto board directamente.
        Más adelante lo convertiremos a tensor.
        """
        return self.board

    def legal_moves(self):
        return list(self.board.legal_moves)

    def step(self, move):
        """
        move: objeto chess.Move
        """

        if self.done:
            raise Exception("La partida ya terminó")

        # aplicar movimiento
        self.board.push(move)

        # comprobar fin de partida
        self.done = self.board.is_game_over()

        reward = 0

        if self.done:
            result = self.board.result()

            if result == "1-0":
                reward = 1
            elif result == "0-1":
                reward = -1
            else:
                reward = 0

        return self.get_state(), reward, self.done, {}