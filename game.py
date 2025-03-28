#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Basic implementation of the game Gobang.

@author: Kingen
"""
import numpy as np
from numpy import ndarray
from scipy.signal import convolve2d

SIZE = 15
WIN_SIZE = 5

EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = -1
SYMBOLS = ['.', 'X', 'O']
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
KERNELS = [
    np.ones((1, WIN_SIZE), dtype=np.int8),
    np.ones((WIN_SIZE, 1), dtype=np.int8),
    np.eye(WIN_SIZE, dtype=np.int8),
    np.fliplr(np.eye(WIN_SIZE, dtype=np.int8))
]


def create_board() -> ndarray:
    """Creates an empty board with the given size."""
    return np.full((SIZE, SIZE), EMPTY, dtype=np.int8)


def print_board(board: ndarray):
    """Prints the board to the console for visualization."""
    print("   " + " ".join(f"{i:2}" for i in range(SIZE)))
    for i in range(SIZE):
        print(f'{i:2}  ' + '  '.join(SYMBOLS[cell] for cell in board[i]))


def is_valid_move(board: ndarray, row, col):
    """Checks if the given position is a valid move."""
    return 0 <= row < SIZE and 0 <= col < SIZE and board[row, col] == EMPTY


def count_consecutive(board: ndarray, row, col, player, dr, dc):
    """Counts consecutive pieces for a player starting from (row, col) in a given direction."""
    count = 0
    while 0 <= row < SIZE and 0 <= col < SIZE and board[row, col] == player:
        count += 1
        row += dr
        col += dc
    return count


def check_win(board: ndarray, row, col, player):
    """Checks if placing a piece at (row, col) by the player results in a win."""
    for dr, dc in DIRECTIONS:
        count = (count_consecutive(board, row, col, player, dr, dc) +
                 count_consecutive(board, row, col, player, -dr, -dc) - 1)
        if count >= WIN_SIZE:
            return True
    return False


def check_win_convolution(board: ndarray, player):
    """Checks if placing a piece at (row, col) by the player results in a win using 2D convolution."""
    player_board = (board == player).astype(np.int8)
    for kernel in KERNELS:
        conv_result: ndarray = convolve2d(player_board, kernel, mode='valid')
        if np.any(conv_result >= WIN_SIZE):
            return True
    return False


def move_manual(board, player):
    while True:
        try:
            row = int(input(f"Player {player} - Enter row (0-{SIZE - 1}): "))
            col = int(input(f"Player {player} - Enter col (0-{SIZE - 1}): "))
        except ValueError:
            print("Invalid input. Please enter numbers.")
            continue
        if not is_valid_move(board, row, col):
            print("Invalid move. Try again.")
            continue
        return row, col


def game_loop(move_one, move_two):
    board = create_board()
    max_moves = SIZE * SIZE

    move_count = 0
    player = PLAYER_ONE
    print_board(board)
    while move_count < max_moves:
        if player == PLAYER_ONE:
            row, col = move_one(board, player)
        else:
            row, col = move_two(board, player)

        board[row, col] = player
        move_count += 1
        print_board(board)

        if check_win_convolution(board, player):
            print(f"Player {player} wins!")
            break
        player = PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE


if __name__ == '__main__':
    game_loop(move_manual, move_manual)
