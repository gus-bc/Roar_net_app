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
from copy import deepcopy
from time import perf_counter

from ..api.Operations import *
from ..api.custom_logger import *


def iterated_local_search(problem: Problem, current_solution: Solution, max_iterations: int, perturbation_strength: int, logger: MetricLogger = None):
    """
    Performs Iterated Local Search (ILS) on the given problem.

    :param problem: The problem instance.
    :param solution: Initial complete solution.
    :param max_iterations: Maximum number of iterations.
    :param perturbation_strength: Number of random destruction moves to apply in perturbation.
    :return: The best found solution.
    """
    logging.info("Starting ILS")
    logging.info(
        f"Max Iterations: {max_iterations} - Perturbation Strength: {perturbation_strength}")
    start_time = perf_counter()

    best_solution = copy_solution(current_solution)
    best_objective = objective_value(best_solution)
    try:
        for iteration in range(1, max_iterations+1):
            current_solution = perturbation(problem, current_solution, perturbation_strength)
            current_objective = objective_value(current_solution)
            improvement = True
            while improvement:
                improvement = False
                best_move = None
                best_improvement = 0
                _moves = moves(local_neighbourhood(problem), current_solution)
                for move in _moves:
                    delta = objective_value_increment(move, current_solution)
                    if delta is not None and delta < best_improvement:
                        best_improvement = delta
                        best_move = move

                if best_move:
                    improvement = True
                    current_solution = apply(best_move, current_solution)
                    current_objective += best_improvement

                cur_colours_used = current_solution.get_num_colours()
                best_colours_used = best_solution.get_num_colours()
                _time = perf_counter() - start_time
                logging.info(
                    f"iteration: {iteration}, current_objective: {current_objective}, current #Colours: {cur_colours_used}, best_objective: {best_objective}, best #Colours: {best_colours_used}, time: {_time}")
                if logger:
                    logger.log_algo_metric(iteration, current_objective,
                                       best_objective,
                                       _time,
                                       cur_colours_used, best_colours_used)

            if current_objective < best_objective:
                best_solution = copy_solution(current_solution)
                best_objective = deepcopy(current_objective)

    except KeyboardInterrupt:
        logging.warning("Manual stop detected! Returning best solution found so far.")
    return best_solution


def perturbation(problem: Problem, solution: Solution, strength: int):
    for _ in range(strength):
        move = random_move(destruction_neighbourhood(problem), solution)
        if move:
            solution = apply(move, solution)
    while True:
        construction_move = random_move(construction_neighbourhood(problem), solution)
        if construction_move:
            solution = apply(construction_move, solution)
        else:
            break
    return solution
