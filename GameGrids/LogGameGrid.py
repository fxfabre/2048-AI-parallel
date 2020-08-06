#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import constants
import itertools

from GameGrids.baseGrid2048 import BaseGrid2048, array2DEquals


class GameGrid2048(BaseGrid2048):

    def clone(self):
        return GameGrid2048(matrix=self.matrix.copy())

    ###############
    # Moves
    ###############
    def moveTo(self, direction=""):
        if self.matrix.max() >= constants.GRID_MAX_VAL:
            return 0, False

        direction = direction.lower()
        if direction == 'left':
            return self.moveLeft()
        elif direction == 'right':
            return self.moveRight()
        elif direction == 'up':
            return self.moveUp()
        elif direction == 'down':
            return self.moveDown()
        return 0, False

    def moveLeft(self):
        _at = self.matrix
        score = 0
        have_moved = False

        # Step 1 : fusion of equal tiles
        for _row in range(self.rows):
            for _column in range(self.columns - 1):

                # find same tile
                for _column_next in range(_column + 1, self.columns):
                    if _at[_row, _column_next] == 0:
                        continue
                    if _at[_row, _column] == _at[_row, _column_next]:
                        self.matrix[_row, _column] += 1
                        self.matrix[_row, _column_next] = 0
                        score += 1 << _at[_row, _column]
                        have_moved = True
                    break

        # Step 2 : Move tiles
        for _row in range(self.rows):

            # Skip columns with > 0 tile
            _first_empty_column = self.columns # après la dernière colonne
            for _column in range(self.columns - 1):
                if _at[_row, _column] == 0:
                    _first_empty_column = _column
                    break

            # Move tiles
            for _column in range(_first_empty_column + 1, self.columns):
                if _at[_row, _column] > 0:
                    self.matrix[_row, _first_empty_column] = _at[_row, _column]
                    _at[_row, _column] = 0
                    _first_empty_column += 1
                    have_moved = True

        return score, have_moved

    def moveRight(self):
        _at = self.matrix
        score = 0
        have_moved = False

        # Step 1 : fusion of equal tiles
        for _row in range(self.rows):
            for _column in range(self.columns - 1, 0, -1):

                # find same tile
                for _column_next in range(_column - 1, -1, -1):
                    if _at[_row, _column_next] == 0:
                        continue
                    if _at[_row, _column] == _at[_row, _column_next]:
                        self.matrix[_row, _column] += 1
                        self.matrix[_row, _column_next] = 0
                        score += 1 << _at[_row, _column]
                        have_moved = True
                    break

        # Step 2 : Move tiles
        for _row in range(self.rows):

            # Skip columns with > 0 tile
            _first_empty_column = -1 # avant la premiere colonne
            for _column in range(self.columns - 1, -1, -1):
                if _at[_row, _column] == 0:
                    _first_empty_column = _column
                    break

            # Move tiles
            for _column in range(_first_empty_column-1, -1, -1):
                if _at[_row, _column] > 0:
                    self.matrix[_row, _first_empty_column] = _at[_row, _column]
                    _at[_row, _column] = 0
                    _first_empty_column -= 1
                    have_moved = True

        return score, have_moved

    def moveUp(self):
        _at = self.matrix
        score = 0
        have_moved = False

        # Step 1 : fusion of equal tiles
        for _column in range(self.columns):
            for _row in range(self.rows - 1):

                # find same tile
                for _row_next in range(_row + 1, self.rows):

                    if _at[_row_next, _column] == 0:
                        continue
                    if _at[_row, _column] == _at[_row_next, _column]:
                        self.matrix[_row, _column] += 1
                        self.matrix[_row_next, _column] = 0
                        score += 1 << _at[_row, _column]
                        have_moved = True
                    break

        # Step 2 : Move tiles
        for _column in range(self.columns):

            # Skip columns with > 0 tile
            _first_empty_row = self.rows # après la dernière colonne
            for _row in range(self.rows - 1):
                if _at[_row, _column] == 0:
                    _first_empty_row = _row
                    break

            # Move tiles
            for _row in range(_first_empty_row+1, self.rows):
                if _at[_row, _column] > 0:
                    self.matrix[_first_empty_row, _column] = _at[_row, _column]
                    _at[_row, _column] = 0
                    _first_empty_row += 1
                    have_moved = True

        return score, have_moved

    def moveDown(self):
        _at = self.matrix
        score = 0
        have_moved = False

        # Step 1 : fusion of equal tiles
        for _column in range(self.columns):
            for _row in range(self.rows - 1, 0, -1):

                # find same tile
                for _row_next in range(_row - 1, -1, -1):

                    if _at[_row_next, _column] == 0:
                        continue
                    if _at[_row, _column] == _at[_row_next, _column]:
                        self.matrix[_row, _column] += 1
                        self.matrix[_row_next, _column] = 0
                        score += 1 << _at[_row, _column]
                        have_moved = True
                    break

        # Step 2 : Move tiles
        for _column in range(self.columns):

            # Skip columns with > 0 tile
            _first_empty_row = -1 # avant la premiere ligne
            for _row in range(self.rows - 1, -1, -1):
                if _at[_row, _column] == 0:
                    _first_empty_row = _row
                    break

            # Move tiles
            for _row in range(_first_empty_row-1, -1, -1):
                if _at[_row, _column] > 0:
                    self.matrix[_first_empty_row, _column] = _at[_row, _column]
                    _at[_row, _column] = 0
                    _first_empty_row -= 1
                    have_moved = True

        return score, have_moved

    ###############
    # Can move
    ###############
    def canMergeLeftRight(self):
        for row in range(self.rows):
            for column in range(self.columns -1):
                if self.matrix[row, column] == 0:
                    pass
                elif self.matrix[row, column] == self.matrix[row, column +1]:
                    return True
        return False

    def canMergeUpDown(self):
        for column in range(self.columns):
            for row in range(self.rows -1):
                if self.matrix[row, column] == 0:
                    pass
                elif self.matrix[row, column] == self.matrix[row +1, column]:
                    return True
        return False

    def canMove(self, direction):
        if direction == 'left':
            return self.canMoveLeft()
        if direction == 'right':
            return self.canMoveRight()
        if direction == 'up':
            return self.canMoveUp()
        if direction == 'down':
            return self.canMoveDown()

        raise Exception("Unknown direction " + str(direction))

    def canMoveLeft(self):
        # Find a tile with an empty box at its left
        for row in range(self.rows):
            for column in range(self.columns -1):
                if (self.matrix[row, column] == 0) and (self.matrix[row, column +1] > 0):
                    return True
        return self.canMergeLeftRight()

    def canMoveUp(self):
        # Find a tile with an empty box above
        for column in range(self.columns):
            for row in range(1, self.rows):
                if (self.matrix[row, column] > 0) and (self.matrix[row -1, column] == 0):
#                    print("({0}, {1}) = 0 and ({2}, {3}) = {4}".format(row, column, row-1, column, self.matrix[row -1, column]))
                    return True
#        print("No tile to move")
        return self.canMergeUpDown()

    def canMoveDown(self):
        # Find a tile with an empty box below
        for column in range(self.columns):
            for row in range(self.rows -1):
                if (self.matrix[row, column] > 0) and (self.matrix[row +1, column] == 0):
                    return True
        return self.canMergeUpDown()

    def canMoveRight(self):
        # Find a tile with an empty box at its right
        for row in range(self.rows):
            for column in range(1, self.columns):
                if (self.matrix[row, column] == 0) and (self.matrix[row, column -1] > 0):
                    return True
        return self.canMergeLeftRight()

    ################
    def is_full(self):
        for i in range(self.rows):
            for j in range(self.columns):
                if self.matrix[i,j] == 0:
                    return False
        return True

    def is_game_over(self):
        if not self.is_full():
            return False

        # Find 2 consecutive and identical tile
        for _row in range(self.rows - 1):
            for _col in range(self.columns - 1):
                if self.matrix[_row, _col] == self.matrix[_row, _col + 1]:
                    return False
                if self.matrix[_row, _col] == self.matrix[_row + 1, _col]:
                    return False
            if self.matrix[_row, self.columns-1] == self.matrix[_row + 1, self.columns-1]:
                return False
        for _col in range(self.columns - 1):
            if self.matrix[self.rows-1, _col] == self.matrix[self.rows -1, _col +1]:
                return False
        return True

    def canAddTile(self, x, y):
        return self.matrix[x,y] == 0

    def addTile(self, x, y, tileToAdd):
        if self.canAddTile(x, y):
            self.matrix[x, y] = tileToAdd
        else:
            print("Unable to add tile at ({0}, {1})".format(x, y))

    def add_random_tile(self):
        """
            pops up a random tile at a given place;
        """
        # ensure we yet have room in grids
        if self.is_full():
            return

        _value = random.choice([1, 1, 1, 2, 1, 1, 1, 1, 1, 1])
        _row, _column = self.get_available_box()
        self.set_tile(_row, _column, _value)

    def set_tile(self, row, col, value):
        if self.matrix[row, col] != 0:
            raise Exception("Tile not empty at ({0}, {1}). can't add tile {2}".format(row, col, value))
        self.matrix[row, col] = value

    def get_available_box (self):
        """
            looks for an empty box location;
        """

        available_box = []
        for _row in range(self.rows):
            for _column in range(self.columns):
                if self.matrix[_row, _column] == 0:
                    available_box.append(_row * self.columns + _column)

        if len(available_box) == 0:
            raise Exception("no more room in grid")

        random_pos = random.choice(available_box)
        return random_pos // self.columns, random_pos % self.columns

    def toIntMatrix(self):
        return self.matrix

    @staticmethod
    def getFinalStates():
        grid_size = constants.NB_ROWS * constants.NB_COLS
        i = 0
        for grid_values in itertools.product(range(1, constants.GRID_MAX_VAL), repeat=grid_size):
            i += 1
            if i % 10000 == 0:
                print("Get final state number", i)

            matrix = np.array(grid_values).reshape(constants.NB_ROWS, constants.NB_COLS)
            grid = GameGrid2048(matrix=matrix)

            if grid.canMergeUpDown():
                continue
            if grid.canMergeLeftRight():
                continue

            yield grid

    def print(self, log_level):
        if self.logger.isEnabledFor(log_level):
            print(self)

    def __str__(self):
        state_val = 0
        for i in range(self.columns):
            for j in range(self.rows):
                state_val = state_val * self.max_value + self.matrix[i, j]

        realValues = np.zeros([self.rows, self.columns], dtype=int)
        for i in range(self.rows):
            for j in range(self.columns):
                realValues[i, j] = 1 << self.matrix[i, j]
        return str(realValues).replace('[1 ', '[. ').replace(' 1 ', ' . ').replace(' 1]', ' .]') + '  ' + str(state_val)

    def __eq__(self, other):
        return array2DEquals(self.matrix, other.matrix)

