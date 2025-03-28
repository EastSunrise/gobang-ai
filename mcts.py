#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Monte Carlo Tree Search algorithm to find the best move.

@author: Kingen
"""

from game import *

CENTER = SIZE // 2


def get_possible_moves_bounding_box(board: ndarray, margin=2):
    """Returns empty positions within the bounding box of occupied cells extended by a margin."""
    occupied = np.nonzero(board != EMPTY)
    if occupied[0].size == 0:
        # empty board
        return [(r, c) for r in range(SIZE) for c in range(SIZE)]

    min_r = max(0, occupied[0].min() - margin)
    max_r = min(SIZE - 1, occupied[0].max() + margin)
    min_c = max(0, occupied[1].min() - margin)
    max_c = min(SIZE - 1, occupied[1].max() + margin)

    sub_board = board[min_r:max_r + 1, min_c:max_c + 1]
    empties = np.argwhere(sub_board == EMPTY)
    return [(r + min_r, c + min_c) for r, c in empties]


def get_possible_moves_radius(board, radius=2):
    """Returns empty positions within a radius of occupied cells."""
    moves = set()
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != EMPTY:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == EMPTY:
                            moves.add((nr, nc))
    return list(moves)


def count_max_consecutive(board, move, player):
    """Returns the max number of consecutive pieces that would be connected."""
    max_total = 0
    row, col = move
    for dr, dc in DIRECTIONS:
        count1 = count_consecutive(board, row + dr, col + dc, player, dr, dc)
        count2 = count_consecutive(board, row - dr, col - dc, player, -dr, -dc)
        total = count1 + count2  # if you place here, you'll connect these pieces
        max_total = max(max_total, total)
    return max_total


def heuristic_score(board, move, player, weight_center=1.0, weight_offensive=10.0, weight_defensive=10.0):
    """Calculates the heuristic score for the given move based on the current board state."""
    row, col = move
    center_score = - abs(CENTER - col) - abs(CENTER - row)

    offensive_total, defensive_total = 0, 0
    opponent = PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE
    for dr, dc in DIRECTIONS:
        count1 = count_consecutive(board, row + dr, col + dc, player, dr, dc)
        count2 = count_consecutive(board, row - dr, col - dc, player, -dr, -dc)
        offensive_total += count1 + count2

        opp_count1 = count_consecutive(board, row + dr, col + dc, opponent, dr, dc)
        opp_count2 = count_consecutive(board, row - dr, col - dc, opponent, -dr, -dc)
        defensive_total += opp_count1 + opp_count2

    offensive_bonus = offensive_total ** 2
    defensive_bonus = offensive_total ** 2

    return weight_center * center_score + weight_offensive * offensive_bonus - weight_defensive * defensive_bonus


def get_expandable_moves(board, player, margin=2, topn=30):
    """Generate moves within a bounding box, score them with multiple heuristics, and return the top N moves."""
    moves = get_possible_moves_bounding_box(board, margin=margin)
    moves.sort(key=lambda x: heuristic_score(board, x, player), reverse=True)
    return moves[:topn]


class Node:
    def __init__(self, board: ndarray, parent=None, move=None, player=PLAYER_ONE):
        self.board = board
        self.parent: Node = parent
        self.move = move  # the move that led to this node
        self.player = player  # the player whose turn it is
        self.opponent = PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE
        self.wins = 0
        self.visits = 0
        self.children = []
        self.untried_moves = get_expandable_moves(board, player)  # untried children (moves)

    def uct_value(self, exploration=1.414):
        """Calculates the UCT (Upper Confidence Bound for Trees) value of this node."""
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        return self.wins / self.visits + exploration * np.sqrt(np.log(parent_visits) / self.visits)

    def select(self):
        """Selects the child node with the highest UCT value."""
        return max(self.children, key=lambda c: c.uct_value())

    def expand(self):
        """Expands one of the untried moves."""
        move = self.untried_moves.pop()
        next_board = self.board.copy()
        next_board[move[0], move[1]] = self.player
        child_node = Node(next_board, parent=self, move=move, player=self.opponent)
        self.children.append(child_node)
        return child_node

    def simulate(self):
        """Simulates a random play-out using weighted random move selection."""
        board = self.board.copy()
        current_player = self.player
        while True:
            empties = np.argwhere(board == EMPTY)
            if empties.size == 0:
                return 0  # draw
            idx = np.random.choice(len(empties))
            row, col = tuple(empties[idx])
            board[row, col] = current_player
            if check_win(board, row, col, current_player):
                return 1 if current_player == self.player else -1
            current_player = PLAYER_TWO if current_player == PLAYER_ONE else PLAYER_ONE

    def backpropagation(self, result):
        """Updates the wins and visits of this node and its ancestors."""
        self.wins += result
        self.visits += 1
        if self.parent:
            self.parent.backpropagation(-result)

    def best_move(self):
        """Returns the move with the highest number of visits."""
        return max(self.children, key=lambda c: c.visits).move


def mcts(board: ndarray, player, iterations=1000):
    """Runs MCTS starting from the given board state and player."""
    root_node = Node(board.copy(), player=player)

    for _ in range(iterations):
        node = root_node

        # selection: traverse the tree using UCT until a node with untried moves is reached
        while not node.untried_moves and node.children:
            node = node.select()

        # expansion: expand one of the untried moves if the node is not terminal
        if node.untried_moves:
            node = node.expand()

        # simulation: simulate a random play-out from the expanded node's state
        result = node.simulate()

        # backpropagation: update the wins and visits of all nodes in the path from the root to the expanded node
        node.backpropagation(result)

    # return the move that has the most visits
    return root_node.best_move()


def move_mcts(board, player):
    """Makes a move using the mcts algorithm."""
    print('AI (mcts) is thinking...')
    return mcts(board, player, 1000)


if __name__ == '__main__':
    game_loop(move_mcts, move_manual)
