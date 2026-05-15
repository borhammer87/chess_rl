from chess_rl.env.chess_env import ChessEnv
from chess_rl.agents.random_agent import RandomAgent


def main():
    env = ChessEnv()
    agent = RandomAgent()

    state = env.reset()

    move_count = 0

    print("=== PARTIDA INICIADA ===")
    print(env.board)
    print()

    while not env.done:
        legal_moves = env.legal_moves()

        move = agent.select_move(legal_moves)

        print(f"Movimiento {move_count + 1}: {move}")

        state, reward, done, info = env.step(move)

        print(env.board)
        print()

        move_count += 1

    print("=== PARTIDA TERMINADA ===")
    print(f"Resultado: {env.board.result()}")
    print(f"Reward final: {reward}")
    print(f"Movimientos totales: {move_count}")


if __name__ == "__main__":
    main()