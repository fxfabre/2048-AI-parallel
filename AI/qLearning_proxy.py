#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pandas
import random
import logging
import traceback

from GameGrids.LogGameGrid import GameGrid2048
from AI.ai_q_learning import Qlearning
from multiprocessing import Queue, Pipe
import constants


EPSILON = 0.2       # 1 means move at random


def start_process_ai(queue : Queue):
    print("start process AI")
    file_pattern = os.path.join(constants.SAVE_DIR, '{0}_{1}'.format(constants.NB_ROWS, constants.NB_COLS))

    ai = Qlearning(queue)
    try:
        ai.init(constants.NB_ROWS, constants.NB_COLS, constants.GRID_MAX_VAL, file_pattern)
        while True:
            ai.run()
            if ai.nb_moves - 1 % 500000 == 0:
                ai.save_states(file_pattern)
            if ai.nb_moves % 1000 == 0:
                print(ai.nb_moves, " moves done")
    except Exception as e:
        print("Stopping AI")
        print(e)
        traceback.print_tb(e.__traceback__)
    ai.save_states(file_pattern)


class QlearningProxy:

    def __init__(self, queue : Queue):
        self._logger = logging.getLogger(__name__)
        self._logger.info("Init Q learning proxy")
        self._moves_list = ['left', 'right', 'up', 'down']

        self.epsilon = EPSILON

        self.id = os.getpid()
        self.queue = queue
        self.parent_conn, self.child_conn = Pipe()
        self.queue.put([self.id, {'conn' : self.child_conn}])
        self.q_val_diff = 0

    def GetMove(self, current_grid : GameGrid2048, history):
        available_moves = [move for move in self._moves_list if current_grid.canMove(move)]
        current_state = self.GetState(current_grid)

        if len(available_moves) == 1:
            return available_moves[0]       # Don't waste time running AI
        if len(available_moves) == 0:
            return self._moves_list[0]      # whatever, it wont't move !

        if (self.epsilon > 0) and (random.uniform(0, 1) < self.epsilon):
            return random.choice(available_moves)

        q_values = self.get_q_values(current_state)
        # print("Q val", q_values)
        # q_values = self.q_values.iloc[current_state, :]

        current_q_val = q_values[available_moves]
        max_val = current_q_val.max()
        optimal_moves = current_q_val[current_q_val == max_val].index.tolist()
        # print("optimal moves", optimal_moves)

        if len(optimal_moves) == 1:
            return optimal_moves[0]
        if len(optimal_moves) == 0:     # shouldn't happen
            raise Exception("No optimal move in Get move function")
        return random.choice(optimal_moves)

    def RecordState(self, s, s_prime, move_dir):
        if move_dir == '':
            move_dir = random.choice(self._moves_list)

        self.queue.put([self.id, {'from' : s, 'to' : s_prime, 'move' : move_dir}])
        return self.q_val_update()

    def get_q_values(self, state : int):
        self.queue.put([self.id, {'state' : state}])
        while True:
            response = self.parent_conn.recv()
            # print("Received response", response)
            if isinstance(response, pandas.Series):
                return response
            self.q_val_diff += response

    def GetState(self, grid):
        total = 0
        for i in range(grid.columns):
            for j in range(grid.rows):
                total = total * grid.max_value + grid.matrix[i, j]
        return total

    def q_val_update(self):
        diff = self.q_val_diff
        self.q_val_diff = 0
        return diff

