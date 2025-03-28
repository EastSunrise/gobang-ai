#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Functions for evaluating the board state.

@author: Kingen
"""
from numpy import ndarray

from game import WIN_SIZE, EMPTY, SIZE, PLAYER_ONE, PLAYER_TWO

SCORES = [
    [0 for i in range(WIN_SIZE)],
    [10 ** (i - 1) for i in range(WIN_SIZE)],  # open_ends=1
    [10 ** i for i in range(WIN_SIZE)],  # open_ends=2
]
WIN_SCORE = 10 ** WIN_SIZE


def eval_segment(count, open_ends):
    """Evaluates a line segment on the count of consecutive pieces and open ends."""
    if count >= WIN_SIZE:
        return WIN_SCORE
    if count <= 1:
        return 0
    return SCORES[open_ends][count]


def eval_line(get_value, length, player):
    """Evaluates a single line (row, column, or diagonal) for segments of the player's pieces."""
    score = 0
    count, open_ends = 0, 0

    for i in range(length):
        current = get_value(i)
        if current == player:
            count += 1
        elif count > 0:  # Check if the segment was open on the before side.
            if i - count - 1 >= 0 and get_value(i - count - 1) == EMPTY:
                open_ends += 1
            if current == EMPTY:
                open_ends += 1
            score += eval_segment(count, open_ends)
            count, open_ends = 0, 0
    if count > 0:  # Evaluate the last segment.
        if length - count - 1 >= 0 and get_value(length - count - 1) == EMPTY:
            open_ends += 1
        score += eval_segment(count, open_ends)
    return score


def eval_row(board: ndarray, r, player):
    """Evaluates a single row for segments of the player's pieces."""
    return eval_line(lambda i: board[r, i], SIZE, player)


def eval_col(board: ndarray, c, player):
    """Evaluates a single column for segments of the player's pieces."""
    return eval_line(lambda i: board[i, c], SIZE, player)


def eval_slash(board: ndarray, r, c, length, player):
    """Evaluates a single slash diagonal for segments of the player's pieces."""
    return eval_line(lambda i: board[r - i, c + i], length, player)


def eval_backslash(board: ndarray, r, c, length, player):
    """Evaluates a single backslash diagonal for segments of the player's pieces."""
    return eval_line(lambda i: board[r + i, c + i], length, player)


def eval_board(board: ndarray, player):
    """Evaluates the board for the perspective of the given player."""
    opponent = PLAYER_ONE if player == PLAYER_TWO else PLAYER_TWO
    score = 0

    for r in range(SIZE):
        score += eval_row(board, r, player) - eval_row(board, r, opponent)
    for c in range(SIZE):
        score += eval_col(board, c, player) - eval_col(board, c, opponent)
    for r in range(SIZE):
        score += eval_slash(board, r, 0, r + 1, player) - eval_slash(board, r, 0, r + 1, opponent)
        score += eval_backslash(board, r, 0, SIZE - r, player) - eval_backslash(board, r, 0, SIZE - r, opponent)
    for c in range(1, SIZE):
        score += eval_slash(board, SIZE - 1, c, SIZE - c, player) - eval_slash(board, SIZE - 1, c, SIZE - c, opponent)
        score += eval_backslash(board, 0, c, SIZE - c, player) - eval_backslash(board, 0, c, SIZE - c, opponent)
    return score
