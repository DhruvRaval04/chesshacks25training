from rl_chess import get_chess_evaluation

# Starting position in FEN. can be replaced with any valid FEN string.
fen = "8/1P1R4/n1r2B2/3Pp3/1k4P1/6K1/Bppr1P2/2q5 w - - 0 1"



evaluation = get_chess_evaluation(fen)

if evaluation:
    print("Evaluation:", evaluation)