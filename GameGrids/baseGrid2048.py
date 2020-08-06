#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import logging
import constants


class BaseGrid2048:
    def __init__(self, nb_rows=0, nb_columns=0, matrix=None):
        self.logger = logging.getLogger(__name__)
        self.isGameOver = False

        if matrix is not None:
            self.matrix = np.array(matrix, dtype=int)
        else:
            self.matrix = np.zeros([nb_rows, nb_columns], dtype=int)
            self.add_random_tile()
            self.add_random_tile()

    def add_random_tile(self):
        raise NotImplementedError()

    @property
    def max_value(self):
        return constants.GRID_MAX_VAL or (self.rows + self.columns + 1)

    @property
    def totalScore(self):
        return self.matrix.sum()

    @property
    def rows(self):
        if self.matrix is None:
            return 0
        return self.matrix.shape[0]

    @property
    def columns(self):
        if self.matrix is None:
            return 0
        return self.matrix.shape[1]


def array2DEquals(matrix_a, matrix_b):
    if matrix_a.shape != matrix_b.shape:
        return False
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_a.shape[1]):
            if matrix_a[i,j] != matrix_b[i,j]:
                return False
    return True

