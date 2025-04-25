# Copyright (C) 2025 Gustav Bech Christensen <gchri21@student.sdu.dk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import logging
import math
from copy import deepcopy
from random import randint
from time import perf_counter
from ..api.custom_logger import MetricLogger
from ..api.Operations import *
from typing import Dict, Optional, Set


def tabu_search(
        problem: Problem,
        current_solution: Solution,
        min_tabu_tenure: int,
        max_tabu_tenure: int,
        max_iterations: int,
        aspiration_criteria: float,
        clean_iter: int,
        idle_iterations: int,
        metric_logger: MetricLogger = None
) -> Optional[Solution]:
    """
    Perform Tabu Search to find an optimized solution.

    :param problem: The problem to be solved
    :param current_solution: The instance of the solution improve
    :param min_tabu_tenure: Minimum number of iterations a move remains tabu
    :param max_tabu_tenure: Maximum number of iterations a move remains tabu
    :param max_iterations: Maximum number of iterations
    :param aspiration_criteria: Threshold to accept a move even if tabu
    :param clean_iter: Cleans tabu list every clean_iter'th iteration
    :param idle_iterations: Stops after n'th iteration without improvement

    :return: The best solution found
    """
    try:
        logging.info("Starting Tabu Search")
        logging.info(f"Max Iterations: {max_iterations}, Min Tabu Tenure: {min_tabu_tenure}, Aspiration: {aspiration_criteria}")

        tabu_set: Set[Move] = set()
        tabu_map: Dict[Move, int] = {}
        iteration = 0
        last_improvement = 0
        start_time = perf_counter()

        best_solution = copy_solution(current_solution)
        best_objective = objective_value(best_solution)
        current_objective = deepcopy(best_objective)

        while iteration < max_iterations and last_improvement < idle_iterations:
            tabu_tenure = min(max(min_tabu_tenure, last_improvement//4), max_tabu_tenure) + randint(-3,3)
            best_move = None
            best_move_value = math.inf

            moves_checked = 0
            moves_tabu = 0
            _moves = moves(local_neighbourhood(problem), current_solution)

            for move in _moves:
                move_value = objective_value_increment(move, current_solution)
                moves_checked += 1
                # Allow move if it's not tabu, its tabu tenure expired, or it meets aspiration criteria
                tabu = not (move not in tabu_set or tabu_map[move] < iteration)
                if tabu:
                    moves_tabu += 1
                if not tabu or move_value <= aspiration_criteria:
                    if move_value < best_move_value:
                        best_move = move
                        best_move_value = move_value

                        # Early exit if this move improves the overall best known objective
                        if current_objective + move_value < best_objective:
                            logging.info(f"Early exit - Best_move: {best_move} - move_value: {move_value}")
                            break

            if not best_move:
                logging.info(f"best_move is None")
                break

            current_solution = apply(best_move, current_solution)
            current_objective += best_move_value

            last_improvement += 1
            if current_objective < best_objective:
                last_improvement = 0
                best_solution = copy_solution(current_solution)
                best_objective = deepcopy(current_objective)

            iteration += 1

            # Update tabu list
            _move = best_move.invert()
            tabu_set.add(_move)
            tabu_map[_move] = iteration + tabu_tenure
            tabu_set.add(best_move)
            tabu_map[best_move] = iteration + tabu_tenure

            # Clean tabu list periodically
            if iteration % clean_iter == 0:
                tabu_map = {m: exp for m, exp in tabu_map.items() if exp > iteration}
                tabu_set.intersection_update(tabu_map.keys())

            # Log to CSV
            time = perf_counter()-start_time
            cur_colours_used = current_solution.get_num_colours()
            best_colours_used =  best_solution.get_num_colours()
            if metric_logger:
                metric_logger.log_algo_metric(iteration, current_objective, best_objective, time, cur_colours_used, best_colours_used)
            # Log to console
            logging.info(f"iteration: {iteration}, last_improvement: {last_improvement}, moves checked: {moves_checked}, tabu_tenure: {tabu_tenure} moves tabu: {moves_tabu}, current_objective: {current_objective}, current #Colours: {cur_colours_used}, best_objective: {best_objective}, best #Colours: {best_colours_used}, time: {time}")
            logging.info(f"Best_move: {best_move} - move_value: {best_move_value}\n")

        if not last_improvement < idle_iterations:
            logging.info(f"Stopped due to idle_iteration reached")
        if not iteration < max_iterations :
            logging.info(f"Stopped due to max_iterations reached")
        return best_solution

    except KeyboardInterrupt:
        logging.warning("Manual stop detected! Returning best solution found so far.")
        return best_solution
    except Exception as e:
        logging.error(str(e))
        return None
