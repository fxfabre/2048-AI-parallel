#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import traceback

from AI.qLearning_proxy import QlearningProxy, start_process_ai
from GameGrids.LogGameGrid import GameGrid2048
from multiprocessing import Queue, Process
import constants

N_PROCESS = 3


def simulate_parallel():
    queue = Queue()

    ai_process = Process(target=start_process_ai, args=(queue,))
    ai_process.start()

    processes = []
    for i in range(N_PROCESS):
        p = Process(target=run_simu, args=(queue, ))
        p.start()
        processes.append(p)

    ai_process.join()


def run_simu(queue):
    game = PlayGame(queue)
    game.simulate()


class PlayGame:
    def __init__(self, queue):
        self._logger = self.init_logger()
        self._ai = QlearningProxy(queue=queue)

    def simulate(self):
        try:
            nb_iter = 0
            while True:
                nb_iter += 1
                self.play_game(nb_iter)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)

    def play_game(self, play_number=0):
        current_grid = GameGrid2048(nb_rows=constants.NB_ROWS, nb_columns=constants.NB_COLS)

        nb_iter = 0
        is_game_over = False
        diff_update = 0
        current_state = self._ai.GetState(current_grid)

        while not is_game_over:
            nb_iter += 1
            self._logger.debug('')
            self._logger.debug("=" * 30)
            self._logger.debug("New loop")
            self._logger.debug("=" * 30)
            current_grid.print(logging.DEBUG)

            old_state = current_state

            move_dir = self._ai.GetMove(current_grid, [])
            current_grid.moveTo(move_dir)
            current_state = self._ai.GetState(current_grid)
            self._logger.debug("Moving %s, from %d to %d", move_dir, old_state, current_state)

            current_grid.print(logging.DEBUG)
            current_grid.add_random_tile()
            current_grid.print(logging.DEBUG)
            current_state = self._ai.GetState(current_grid)

            if current_grid.matrix.max() >= constants.GRID_MAX_VAL:
                # self._logger.info("Stop iterations, values too big for the model")
                is_game_over = True
            else:
                diff_update += self._ai.RecordState(old_state, current_state, move_dir)
                is_game_over = current_grid.is_game_over()

        diff_update += self._ai.q_val_update()
        self._logger.info("{0:>3} Game over in {1:>3} iterations, diff = {2:>9}".format(
            play_number, nb_iter, int(diff_update)))

    def init_logger(self):
        log_filename = os.path.join('/tmp', "2048_" + str(int(time.time())) + ".log")
        log_format = '%(asctime)-15s %(message)s'
        logging.basicConfig(format=log_format, level=logging.INFO)
        return logging.getLogger(__name__)


if __name__ == '__main__':
    simulate_parallel()
