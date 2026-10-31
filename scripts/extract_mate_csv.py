
"""Extract distinct mate-in-one positions from PGN games into a CSV."""

import argparse
import csv
from pathlib import Path

import chess
import chess.pgn


def extract_mates(paths, output, existing=None):
    rows = {}

    if existing is not None:
        with existing.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                board = chess.Board(row["fen"])
                key = (
                    board.board_fen(),
                    board.turn,
                    board.castling_rights,
                    board.ep_square,
                )
                rows.setdefault(
                    key, (row["fen"], row["mate_moves"])
                )

    for path in sorted(paths):
        with path.open(encoding="utf-8") as handle:
            while (game := chess.pgn.read_game(handle)) is not None:
                board = game.board()

                for move in game.mainline_moves():
                    mates = []

                    for candidate in list(board.legal_moves):
                        board.push(candidate)
                        if board.is_checkmate():
                            mates.append(candidate.uci())
                        board.pop()

                    if mates:
                        key = (
                            board.board_fen(),
                            board.turn,
                            board.castling_rights,
                            board.ep_square,
                        )
                        rows.setdefault(
                            key,
                            (board.fen(), " ".join(mates)),
                        )

                    board.push(move)

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("fen", "mate_moves"))
        writer.writerows(rows.values())

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--existing", type=Path, default=None
    )
    args = parser.parse_args()

    paths = list(args.pgn_directory.glob("*.pgn"))

    if not paths:
        parser.error("No PGN files found")

    count = extract_mates(
        paths, args.output, args.existing
    )
    print(f"Extracted {count} unique mate positions")


if __name__ == "__main__":
    main()
