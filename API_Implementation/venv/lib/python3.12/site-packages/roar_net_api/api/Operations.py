from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Iterable, TypeVar, List


class ProblemInterface(ABC):
    @abstractmethod
    def construction_neighbourhood(self: Problem) -> Neighbourhood:
        ...

    @abstractmethod
    def destruction_neighbourhood(self: Problem) -> Neighbourhood:
        ...

    @abstractmethod
    def local_neighbourhood(self: Problem) -> Neighbourhood:
        ...

    @abstractmethod
    def random_solution(self: Problem) -> Solution:
        ...

    @abstractmethod
    def empty_solution(self: Problem) -> Solution:
        ...

    @abstractmethod
    def heuristic_solution(self: Problem) -> Optional[Solution]:
        ...


class SolutionInterface(ABC):

    @abstractmethod
    def objective_value(self) -> float:
        ...

    @abstractmethod
    def copy_solution(self: Solution) -> Solution:
        ...

    @abstractmethod
    def is_feasible(self: Solution) -> bool:
        ...

    @abstractmethod
    def objective_value_increment(self: Solution, move: Move) -> Optional[float]:
        ...

    @abstractmethod
    def apply(self: Solution, move: Move) -> Solution:
        ...

    @abstractmethod
    def get_num_colours(self: Solution) -> int:
        ...

    @abstractmethod
    def check_colouring_constraints(self: Solution):
        ...

class NeighbourhoodInterface(ABC):
    @abstractmethod
    def moves(self: Neighbourhood, solution: Solution) -> Iterable[Move]:
        ...

    @abstractmethod
    def random_move(self: Neighbourhood, solution: Solution) -> Optional[Move]:
        ...

    @abstractmethod
    def random_moves_without_replacement(self: Neighbourhood, solution: Solution) -> Optional[Move]:
        ...


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
class MoveInterface(ABC):
    @abstractmethod
    def invert(self: Move) -> Move:
        raise NotImplementedError


Solution = TypeVar('Solution', bound=SolutionInterface)
Move = TypeVar('Move', bound=MoveInterface)
Neighbourhood = TypeVar('Neighbourhood', bound=NeighbourhoodInterface)
Problem = TypeVar('Problem', bound=ProblemInterface)


def apply(move: Move, solution: Solution) -> Solution:
    return solution.apply(move)

def construction_neighbourhood(problem: Problem) -> Neighbourhood:
    return problem.construction_neighbourhood()

def destruction_neighbourhood(problem: Problem) -> Neighbourhood:
    return problem.destruction_neighbourhood()

def local_neighbourhood(problem: Problem) -> Neighbourhood:
    return problem.local_neighbourhood()

def empty_solution(problem: Problem) -> Solution:
    return problem.empty_solution()

def random_solution(problem: Problem) -> Solution:
    return problem.random_solution()

def heuristic_solution(problem: Problem) -> Optional[Solution]:
    return problem.heuristic_solution()

def copy_solution(solution: Solution) -> Solution:
    return solution.copy_solution()

def invert(move: Move) -> Move:
    return move.invert()

def is_feasible(solution: Solution) -> bool:
    return solution.is_feasible()

def lower_bound_increment(move: Move, solution: Solution) -> Optional[float]:
    return solution.lower_bound_increment()

def lower_bound(solution: Solution) -> Optional[float]:
    return solution.lower_bound()

def moves(neighbourhood: Neighbourhood, solution: Solution) -> Iterable[Move]:
    return neighbourhood.moves(solution)

def objective_value_increment(move: Move, solution: Solution) -> Optional[float]:
    return solution.objective_value_increment(move)

def objective_value(solution: Solution) -> Optional[float]:
    return solution.objective_value()

def random_move(neighbourhood: Neighbourhood, solution: Solution) -> Optional[Move]:
    return neighbourhood.random_move(solution)

def random_moves_without_replacement(neighbourhood: Neighbourhood, solution: Solution) -> Optional[Move]:
    return neighbourhood.random_moves_without_replacement(solution)
