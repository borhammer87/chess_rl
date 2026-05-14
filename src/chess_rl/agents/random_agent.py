import random


class RandomAgent:
    def select_move(self, legal_moves):
        """
        Selecciona un movimiento aleatorio.
        """
        return random.choice(legal_moves)