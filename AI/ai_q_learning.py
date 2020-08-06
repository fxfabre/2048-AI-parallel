#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import random
import pandas
import logging
import traceback

from AI.ai_base import BaseAi
from GameGrids.LogGameGrid import GameGrid2048
from multiprocessing import Queue
import constants

ALPHA = 0.5
GAMMA = 1.0
EPSILON = 0.2       # 1 means move at random
REWARD_MOVE = 1.0
REWARD_END_GAME = -10.0


class Qlearning(BaseAi):

    def __init__(self, queue : Queue):
        self._logger = logging.getLogger(__name__)
        self._logger.info("Init Q learning")
        self._moves_list = ['left', 'right', 'up', 'down']
        self.nb_moves = 0

        self.q_values = pandas.DataFrame()
        self.epsilon = EPSILON
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.reward_move = REWARD_MOVE
        self.reward_end_game = REWARD_END_GAME

        self.queue = queue
        self.clients = {}

    def init(self, nb_row, nb_col, max_value, file_pattern):
        nb_box      = nb_row * nb_col
        nb_states   = max_value ** nb_box
        self.q_values = pandas.DataFrame(index=range(nb_states), columns=self._moves_list)
        if not self.load_states(file_pattern):
            self.q_values.fillna(0, inplace=True)
            self.init_end_states()
        self._logger.info('Init Q learning success : %s', self.q_values.shape)

    def init_end_states(self):
        self._logger.debug("Start init end states")
        for grid in GameGrid2048.getFinalStates():
            state = self.get_state(grid)
            self.q_values.iloc[state, :] = [REWARD_END_GAME] * 4
        self._logger.debug("Init end states done")

    def run(self):
        query = self.queue.get()
        id = query[0]
        params = query[1]
        # print("AI : received query", id, params)

        if 'conn' in params:
            self.register_client(id, params.get('conn'))

        elif 'state' in params:
            self.send_q_val(id, params.get('state'))

        elif 'from' in params:
            self.record_move(id, params['from'], params['to'], params['move'])

        else:
            print("ERROR : Unknown params", id, params)

        # self.queue.put([self.id, {'conn': self.child_conn}])
        # self.queue.put([self.id, {'from': s, 'to': s_prime, 'move': move_dir}])
        # self.queue.put([self.id, {'state': state}])

    def register_client(self, pid, conn):
        # print("Connection registered for client", pid)
        self.clients[pid] = conn

    def send_q_val(self, pid : int, state : int):
        # conn = multiprocessing.Connection()
        conn = self.clients.get(pid)
        # print("Sending to process", pid, "Q values :", self.q_values.iloc[state, :])
        conn.send(self.q_values.iloc[state, :])

    def record_move(self, pid, s, s_prime, move_dir):
        self.nb_moves += 1

        a = self._moves_list.index(move_dir)
        q_value_s = self.q_values.iloc[s, a]
        q_value_s_prime = self.q_values.iloc[s_prime, :].max()

        self._logger.debug("Update q values from state %s, move %s to state %s", s, move_dir, s_prime)
        self._logger.debug("\n%s", self.q_values.iloc[s, :])

        value_to_add = self.alpha * (self.reward_move + self.gamma * q_value_s_prime - q_value_s)
        self.q_values.iloc[s, a] += value_to_add

        self._logger.debug("Diff : %s => new value %s : %s", value_to_add, s, self.q_values.iloc[s, a])
        self._logger.debug("\n%s", self.q_values.iloc[s_prime, :])

        # print("Sending qval update", value_to_add, "to", pid)
        self.clients[pid].send(abs(value_to_add))    # return value

    def save_states(self, name):
        file_name = name + '_qValues.csv'
        # print("Saving file to %s", file_name)
        self.q_values.to_csv(file_name, sep='|')

    def load_states(self, name):
        file_name = name + '_qValues.csv'
        current_shape = self.q_values.shape
        if os.path.exists(file_name):
            # print("Read Q values file from %s", file_name)
            self.q_values = pandas.read_csv(file_name, sep='|', index_col=0)
            self._moves_list = self.q_values.columns.tolist()
            assert current_shape == self.q_values.shape
            return True
        return False

    def get_state(self, grid):
        total = 0
        for i in range(grid.columns):
            for j in range(grid.rows):
                total = total * grid.max_value + grid.matrix[i, j]
        return total
