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
from time import perf_counter
from ..api.custom_logger import *
from ..api.Operations import *


def best_improvement(problem: Problem, current_solution: Solution, budget: float, logger: MetricLogger = None) -> Solution:

    iteration = 0
    start_time = perf_counter()

    best_solution = copy_solution(current_solution)
    best_objective = objective_value(best_solution)
    time = perf_counter() - start_time
    while time < budget:
        iteration += 1
        neighbourhood: Neighbourhood = local_neighbourhood(problem)
        best_incr = 0
        best_move = None

        for move in moves(neighbourhood, current_solution):
            delta = objective_value_increment(move, current_solution)
            if delta < best_incr:
                best_incr = delta
                best_move = move

        if best_move is None:
            break

        current_solution = apply(best_move, current_solution)
        current_objective = objective_value(current_solution)

        cur_colours_used = current_solution.get_num_colours()
        best_colours_used = best_solution.get_num_colours()

        # Log to CSV
        time = perf_counter() - start_time
        if logger:
            logger.log_algo_metric(iteration, current_objective, best_objective, time, cur_colours_used, best_colours_used)
        if best_move is not None:
            apply(best_move, current_solution)
        else:
            break
    return best_solution