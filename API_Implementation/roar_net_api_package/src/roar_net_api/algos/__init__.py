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
from .best_improvement import best_improvement
from .iterated_local_search import iterated_local_search
from .tabu_search import tabu_search
from .simulated_annealing import simulated_annealing

__solvers__ = ["tabu_search", "iterated_local_search", "best_improvement", "simulated_annealing"]

__all__ = ["tabu_search", "iterated_local_search", "best_improvement", "simulated_annealing"]