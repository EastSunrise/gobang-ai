#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Minimax algorithm with alpha-beta pruning to find the best move.

@author: Kingen
"""
from evaluation import eval_board, WIN_SCORE
from game import SIZE, is_valid_move, check_win, EMPTY, PLAYER_ONE, PLAYER_TWO, game_loop


def minimax(board, depth, alpha, beta, maximizer, player):
    """
    Minimax algorithm with alpha-beta pruning.
    :param board: current board state
    :param depth: remaining depth
    :param alpha: best value for maximizer so far
    :param beta: best value for minimizer so far
    :param maximizer: boolean indicating if current move is maximizing
    :param player: the player whose turn it is
    :return: a tuple of (score, best_move) where move is (row, col)
    """
    if depth == 0:
        return eval_board(board, player), None

    best_move = None
    opponent = PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE

    # Generate list of all valid moves and order them to help alpha-beta pruning work more efficiently
    moves = []
    for row in range(SIZE):
        for col in range(SIZE):
            if is_valid_move(board, row, col):
                board[row][col] = player if maximizer else opponent
                # use a shallow evaluation to order moves
                score = eval_board(board, player)
                moves.append((row, col, score))
                board[row][col] = EMPTY
    moves.sort(key=lambda x: x[2], reverse=maximizer)

    if maximizer:
        max_score = float('-inf')
        for row, col, _ in moves:
            board[row][col] = player  # make move
            if check_win(board, row, col, player):
                return WIN_SCORE, (row, col)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, False, player)
            board[row][col] = EMPTY  # undo move

            if eval_score > max_score:
                max_score = eval_score
                best_move = (row, col)

            # Update alpha with maximum score found so far
            alpha = max(alpha, eval_score)
            if alpha >= beta:  # prune
                break
        return max_score, best_move
    else:
        min_score = float('inf')
        for row, col, _ in moves:
            board[row][col] = opponent  # make move
            if check_win(board, row, col, opponent):
                return -WIN_SCORE, (row, col)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, True, player)
            board[row][col] = EMPTY  # undo move

            if eval_score < min_score:
                min_score = eval_score
                best_move = (row, col)

            # Update beta with minimum score found so far
            beta = min(beta, eval_score)
            if alpha >= beta:  # prune
                break
        return min_score, best_move


def move_minimax(board, player):
    """Makes a move using the minimax algorithm with alpha-beta pruning."""
    print('AI (minimax) is thinking...')
    _, best_move = minimax(board, 1, float('-inf'), float('inf'), True, player)
    if best_move is None:
        raise Exception('AI cannot make a move')
    return best_move


if __name__ == '__main__':
    game_loop(move_minimax, move_minimax)
