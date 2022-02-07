import numpy as np
from rl import markov_decision_process

from typing import Tuple, Mapping

from rl.distribution import Categorical, Constant
from dataclasses import dataclass

import abc
from rl import dynamic_programming
from rl.midterm_2022 import priority_q 

SPACE = 'SPACE'
BLOCK = 'BLOCK'
GOAL = 'GOAL'

import random


# Create a maze using the depth-first algorithm described at
# https://scipython.com/blog/making-a-maze/
# Christian Hill, April 2017.

class Cell:
    """A cell in the maze.

    A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.

    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    
    def __str__(self):
        return f'{self.x, self.y}'

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())
    
    def has_wall_at(self, direction):
        """Does this cell still have all its walls?"""
        return self.walls[direction]

        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False


class Maze:
    """A Maze, represented as a grid of cells."""

    def __init__(self, nx, ny, ix=0, iy=0):
        """Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).

        """

        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]
        self.make_maze()

    def cell_at(self, x, y):
        """Return the Cell object at (x,y)."""

        return self.maze_map[x][y]

    def __str__(self):
        """Return a (crude) string representation of the maze."""

        maze_rows = self.maze_rows()
        return '\n'.join(maze_rows)
    
    def maze_rows(self):
        """Return a (crude) string representation of the maze."""

        maze_rows = ['-' * self.nx * 2]
        for y in range(self.ny):
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['E']:
                    maze_row.append(' |')
                else:
                    maze_row.append('  ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return maze_rows

    
    def find_valid_neighbours(self, cell):
        """Return a list of neighbours to cell."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)

                neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self):
        # Total number of cells.
        N = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        possible_next_cells = []
        visited = set()
        visited.add(current_cell)
        found_list = [n for _,n in self.find_valid_neighbours(current_cell)]
        found = set(found_list)
        # Total number of visited cells during maze construction.
        nv = 1

        while len(visited) < N:
#             print([str(x) for x in visited], 
#                   [str(x) for x in found_list], 
#                   [str(x) for x in found])
            next_node = random.choice(found_list)
#             print(next_node)
            found.remove(next_node)
            found_list.remove(next_node)
            visited.add(next_node)
            
            candidates = [(d, n) for d, n in self.find_valid_neighbours(next_node) if n in visited]
#             print(candidates)

            direction, chosen_source = random.choice(candidates)
            
            for _,n in self.find_valid_neighbours(next_node):
                if n not in visited:
                    if n not in found:
                        found_list.append(n)
                        found.add(n)
#             print(next_node, direction, chosen_source)
            
            next_node.knock_down_wall(chosen_source, direction)
#             print(self)
            
            
@dataclass(frozen=True)
class GridState:

    x: int
    y: int

    def __lt__(self, other):
        '''Your code here, implement a comparison function that should satisfy'''
        return (self.x, self.y) < (other.x, other.y)
    
    
class GridMazeMDP(markov_decision_process.FiniteMarkovDecisionProcess[GridState, int], abc.ABC):
    def __init__(self, maze: Maze, goal_x, goal_y):
        self.moves = ["Up", "Down", "Left", "Right"]
        self.goal = GridState(goal_x, goal_y)
        self.maze = maze
        
        super().__init__(self.get_action_transition_reward_map(maze))

    def get_action_transition_reward_map(self, maze: Maze):
        d: Dict[GridState, Dict[str, Categorical[Tuple[GridState, float]]]] = {}
        
        for x in range(maze.nx):
            for y in range(maze.ny):            
                state = GridState(x, y)
                if state != self.goal:
                    d1: Dict[str, Categorical[Tuple[GridState, float]]] = {}
                    cell = maze.cell_at(x,y)
                    for move, next_cell in maze.find_valid_neighbours(cell):
                        if not cell.has_wall_at(move):
                            next_state = GridState(next_cell.x, next_cell.y)
                            d1[move] = Constant((next_state, self.reward_func(next_state)))
                    d[state] = d1
        return d

    @abc.abstractmethod
    def reward_func(self, next_state) -> float:
        pass
    
    def __str__(self):
        """Return a (crude) string representation of the maze."""
        maze_rows = self.maze.maze_rows()
        maze_rows[self.goal.x*2 + 1] = maze_rows[self.goal.x*2 + 1][:self.goal.y*2+1] + "*" \
                        + maze_rows[self.goal.x*2 + 1][self.goal.y*2 +2:]
        return '\n'.join(maze_rows)
    
    
    def print_policy(self, policy):
        """Return a (crude) string representation of the maze."""
        maze_rows = self.maze.maze_rows()
        maze_rows[self.goal.x*2 + 1] = maze_rows[self.goal.x*2 + 1][:self.goal.y*2+1] + "*" \
                        + maze_rows[self.goal.x*2 + 1][self.goal.y*2 +2:]
        
        for state in self.non_terminal_states:
            state = state.state
            action = policy.action_for[state]
            if action == 'N':
                str_val = '^'
            elif action == 'E':
                str_val = '>'
            elif action == 'S':
                str_val = 'v'
            elif action == 'W':
                str_val = '<'
            maze_rows[state.y*2 + 1] = maze_rows[state.y*2 + 1][:state.x*2+1] + str_val \
                        + maze_rows[state.y*2 + 1][state.x*2 +2:]
        return '\n'.join(maze_rows)

class GridMazeMDP_Dense(GridMazeMDP):
    def reward_func(self, next_state) -> float:
        return -1
    
class GridMazeMDP_Sparse(GridMazeMDP):
    def reward_func(self, next_state) -> float:
        if next_state == self.goal:
            return 1
        return 0
