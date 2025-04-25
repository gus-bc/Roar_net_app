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
from __future__ import annotations

import cProfile
import copy
import inspect
import os
from collections import defaultdict
from enum import Enum
from random import choice
from typing import TextIO

from roar_net_api.api.Operations import *
from roar_net_api.api.custom_logger import *
from roar_net_api.api.utils import *
from roar_net_api.api.generator_utils import *
from roar_net_api.api.Types import *
from roar_net_api.algos import *



class Problem(ProblemInterface):
    def __init__(self, v: int, edge_list: EdgeList) -> None:
        self.vertex_count = v
        self.adjacency_matrix = adjacency_matrix(v, edge_list)
        self.adjacency_list = adjacency_list(v, edge_list)


    @classmethod
    def from_textio(cls, file: TextIO) -> Problem:
        """
        Reads a problem instance from a .txt file and returns a Problem object encoding the problem instance.

        Parameters
        ----------
        file : TextIO
            A file-like object that supports reading lines, from which the problem instance is read.

        Returns
        -------
        Problem
            A Problem object initialized with the number of vertices and the list of edges read from the file.
        """
        v = int(file.readline())
        edges = []
        for line in file:
            x, y = map(int, line.strip().split(","))
            edges.append((x, y))
        return cls(v, edges)

    @classmethod
    def from_col(cls, file: TextIO):
        """
            Reads a problem instance from a .col file and returns a Problem object encoding the problem instance.

            Parameters
            ----------
            file : TextIO
                A file-like object that supports reading lines, from which the problem instance is read.

            Returns
            -------
            Problem
                A Problem object initialized with the number of vertices and the list of edges read from the file.
            """
        edges = []
        for line in file:
            line = line.strip()
            if line.startswith('p'):
                _, _, num_vertices, num_edges = line.split()
            elif line.startswith('e'):
                _, u, v = line.split()
                edges.append((int(u) - 1, int(v) - 1))

        return cls(int(num_vertices), edges)

    def construction_neighbourhood(self: Problem) -> Construction_neighbourhood:
        return Construction_neighbourhood(self)

    def destruction_neighbourhood(self: Problem) -> Destruction_neighbourhood:
        return Destruction_neighbourhood(self)

    def local_neighbourhood(self: Problem) -> Local_neighbourhood:
        return Local_neighbourhood(self)

    def random_solution(self: Problem) -> Solution:
        """
        Constructs a random solution by iteratively applying random construction moves.

        Starts with an empty solution and sequentially applies a random move
        from the construction neighborhood until all vertices are assigned. The moves
        are chosen from the Construction_neighbourhood of the current partial solution.

        Parameters
        ----------
        self : Problem
            The problem instance that contains the necessary information for constructing a solution.

        Returns
        -------
        Solution
            A fully constructed solution for the given problem.
        """
        solution = Solution(self)
        for v in range(self.vertex_count):
            neighbourhood: Construction_neighbourhood = Construction_neighbourhood(self)
            move: Move = neighbourhood.random_move(solution)
            if move is None:
                pass
            solution.apply(move)

        return solution

    def empty_solution(self: Problem) -> Solution:
        """
        Returns an empty solution for the given problem.

        Parameters
        ----------
        self : Problem
            The problem instance that the empty solution will be constructed for.

        Returns
        -------
        Solution
            An empty solution for the given problem, initialized with no assignments.

        """
        return Solution(self)

    def heuristic_solution(self: Problem) -> Optional[Solution]:
        """
        Constructs a heuristic solution by assigning the lowest colour that doesn't
        introduce an edge conflict for each vertex.

        This method iterates through all vertices and assigns the lowest available
        colour (integer) that doesn't create an edge conflict with already assigned vertices.

        Parameters
        ----------
        self : Problem
            The problem instance that the heuristic solution will be constructed for.

        Returns
        -------
        Optional[Solution]
            A feasible solution if successful, otherwise `None` if an error occurs during
            the colour assignment for any vertex.

        """
        solution = Solution(self)

        vertices = list(range(self.vertex_count))
        random.shuffle(vertices)

        for v in vertices:
            try:
                neighbourhood = Construction_neighbourhood(self)
                _moves = [move for move in neighbourhood.moves(solution) if move.vertex == v and move.new_colour not in solution.adjacent_colours(v)]
                move = min(_moves, key=lambda x: x.new_colour)
                solution.apply(move)
            except ValueError:
                logging.error(f"Vertex {v}: Failed to assign a colour")
                return None
        return solution

class Solution(SolutionInterface):
    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.colour_classes = {}
        self.vertex_colours = {v: 0 for v in range(problem.vertex_count)}
        self.conflicts = defaultdict(lambda: defaultdict(lambda: [0, set()]))

        self.local_move_generator = iter(sample2(self.problem.vertex_count,
                                                       min(len(self.colour_classes.keys()) + 1,
                                                           self.problem.vertex_count)))
        self.construction_move_generator = iter(sample2(self.problem.vertex_count,
                                                              min(len(self.colour_classes.keys()) + 1,
                                                                  self.problem.vertex_count)))
        self.destruction_move_generator = iter(sample(self.problem.vertex_count))

    def __hash__(self):
        return self.vertex_colours.__hash__

    def __repr__(self):
        return f"{self.colour_classes}"

    def objective_value(self) -> Optional[float]:
        """
        Calculates the cost of the solution based on vertex colours and problem constraints.

        The objective value is computed as the difference between two terms:
        1. The negative sum of squared sizes of colour classes.
        2. The sum of twice the product of the size of each colour class and the number of edges between vertices in the class.

        Parameters
        ----------
        self : Solution
            The Solution instance for which the objective value is calculated, based on the colouring of the vertices.

        Returns
        -------
        Optional[float]
            The objective value as a floating-point number. Returns `None` if the objective cannot be computed.

        """

        # First term: - \sum_{i=1}^{k} |C_i|^2
        sum_c_squared = sum(len(nodes) ** 2 for nodes in self.colour_classes.values())

        # Second term: \sum_{i=1}^{k} 2|C_i| \cdot |E_i|
        sum_c_e = 0
        for vertices in self.colour_classes.values():
            edge_count = 0
            for v in vertices:
                for u in self.problem.adjacency_list[v]:
                    if u in vertices and u < v:
                        edge_count += 1
            sum_c_e += 2 * len(vertices) * edge_count

        return -sum_c_squared + sum_c_e


    def objective_value_increment(self: Solution, move: Move) -> Optional[float]:
        """
        Computes the change in the objective value due to a colour change for a given vertex.

        Parameters
        ----------
        self : Solution
            The solution instance in which the objective value change is being calculated.
        move : Move
            A `Move` object that contains the information about the vertex whose colour is being changed. It includes
            the vertex (`vertex`), the old colour (`old_colour`), and the new colour (`new_colour`).

        Returns
        -------
        Optional[float]
            The change in the objective value resulting from the colour change. This value is positive or negative,
            indicating an increase or decrease in the objective value. Returns `None` if the calculation fails.
        """
        # Cache attributes locally (faster than repeated attribute access)
        vertex = move.vertex
        old_colour = move.old_colour
        new_colour = move.new_colour

        old_class = self.colour_classes[old_colour]
        new_class = self.colour_classes.get(new_colour, {})

        old_size = len(old_class)
        new_size = len(new_class)

        # Term 1: Change in sum of squared sizes
        delta_size_term = 2 * ( old_size - new_size - 1)

        # Term 2: Added conflicts in colour_class
        conflict_old = self.conflicts[vertex][old_colour][0]
        conflict_new = self.conflicts[vertex][new_colour][0]

        # Change due to colour_class size difference
        delta_old_colour_class = self._total_conflicts_in_colour_class(old_colour)
        delta_new_colour_class = self._total_conflicts_in_colour_class(new_colour)

        delta_old_edge_term = -2 * ((old_size - 1) * conflict_old + delta_old_colour_class)
        delta_new_edge_term = 2 * ((new_size + 1) * conflict_new + delta_new_colour_class)

        delta_edge_term = delta_old_edge_term + delta_new_edge_term

        return delta_size_term + delta_edge_term


    def lower_bound(self):
        """
        Computes a lower bound by calculation the evaluation of the current colour classes and
        assign uncoloured vertices into the largest colour class to minimize the sum of squares
        term while assuming no new conflicts

        Parameters
           ----------
           self : Solution
               The solution containing current colour classes and problem data.

        Returns
        -------
        float
            A tight lower bound consistent with the objective function.
        """
        # Get uncoloured vertices (those with color 0)
        uncoloured_vertices = {v for v, c in self.vertex_colours.items() if c == 0}
        remaining_vertices = len(uncoloured_vertices)

        # Find the largest colour class
        largest_colour_class = [0,0]

        # Calculate current contribution from coloured vertices
        current_value = 0
        for colour, vertices in self.colour_classes.items():
            class_size = len(vertices)
            if class_size > largest_colour_class[1]:
                largest_colour_class = [colour, class_size]
            conflicts_in_colour_class = self._total_conflicts_in_colour_class(colour)
            current_value += -class_size ** 2 + 2 * class_size * conflicts_in_colour_class

        # Add uncoloured vertices to largest colour class
        value = -(2*largest_colour_class[1]*remaining_vertices + remaining_vertices**2)

        return current_value + value

    def lower_bound_increment(self, move: Move):
        """
        Computes the change in the lower bound due to a potential move.

        Parameters
        ----------
        self : Solution
            The solution containing current colour classes and problem data.
        move : Move
            A `Move` representing an assignment of a colour to a vertex colour.

        Returns
        -------
        Optional[float]
            The change in the lower bound objective.
        """

        vertex = move.vertex
        new_colour = move.new_colour
        new_class = self.colour_classes.get(new_colour, {})
        new_size = len(new_class)
        delta_new_colour_class = self._total_conflicts_in_colour_class(new_colour)

        uncoloured_vertices = {v for v, c in self.vertex_colours.items() if c == 0}
        remaining_vertices = len(uncoloured_vertices)

        largest_colour_class = max(self.colour_classes.items(),
                                   key=lambda item: len(item[1]))[0] if self.colour_classes else 0
        largest_colour_class_size = len(self.colour_classes.get(largest_colour_class, {}))

        conflict_new = self.conflicts[vertex][new_colour][0]
        delta_new_edge_term = 2 * ((new_size + 1) * conflict_new + delta_new_colour_class)

        if new_colour == largest_colour_class or largest_colour_class == 0:
            return delta_new_edge_term
        else:
            value = 2*remaining_vertices - 1
            assign_class_delta = -2 * new_size + 1
            return value + assign_class_delta + delta_new_edge_term

    def copy_solution(self: Solution) -> Solution:
        """
        Creates a deep copy of the current solution and returns the copy.

        Parameters
        ----------
        self : Solution
            The solution instance to be copied.

        Returns
        -------
        Solution
            A new `Solution` object that is a deep copy of the original.

       """
        new_obj = Solution(self.problem)

        new_obj.colour_classes = copy.deepcopy(self.colour_classes)
        new_obj.vertex_colours = copy.deepcopy(self.vertex_colours)
        new_obj.conflicts = copy.deepcopy(self.conflicts)

        new_obj.local_move_generator = sample2(self.problem.vertex_count - 1, min(len(self.colour_classes.keys()) + 1, self.problem.vertex_count))
        new_obj.construction_move_generator = sample2(self.problem.vertex_count - 1, min(len(self.colour_classes.keys()) + 1, self.problem.vertex_count))
        new_obj.destruction_move_generator = sample(self.problem.vertex_count - 1)

        return new_obj

    def is_feasible(self: Solution) -> bool:
        """
        Checks if the solution is feasible.

        A solution is considered feasible if all vertices are assigned a colour (i.e., no vertex
        has a colour of 0). This method checks whether the number of vertices with non-zero colours
        matches the total number of vertices in the problem.

        Parameters
        ----------
        self : Solution
            The solution instance to be checked for feasibility.

        Returns
        -------
        bool
            `True` if the solution is feasible (all vertices have a non-zero colour), otherwise `False`.
        """
        return len([v for v, c in self.vertex_colours.items() if c != 0]) == self.problem.vertex_count


    def apply(self: Solution, move: Move) -> Solution:
        """
        Applies a given move to the solution by calling helper methods.

        Parameters
        ----------
        self : Solution
               The solution instance to which the move is applied.
        move : Move
            A `Move` object that contains the information about the move to be applied. The move specifies the type
            of operation (construction, destruction, or recolouring) and the details of the vertex or colour change.

       Returns
       -------
       Solution
           The updated `Solution` object after the move has been applied.
       """
        solution = None
        match move.move_type:
            case Move_type.CONSTRUCTION_MOVE:
                solution = self._construction_move(move)
            case Move_type.DESTRUCTION_MOVE:
                solution = self._destruction_move(move)
            case Move_type.ONE_EXCHANGE:
                solution = self._one_exchange(move)
            case _:
                # Can't happen
                pass
        if solution is None:
            pass
        solution._reset_move_generator()
        return solution

    # ___ Helper methods: ___
    def get_num_colours(self) -> int:
        """
        Returns the number of colours used.

        Parameters
        ----------
        self : Solution
            The solution instance to which the move is applied.

        Returns
        -------
        int
            The number os colours used.
        """
        return len(self.colour_classes.keys())

    def adjacent_colours(self, v: int) -> set[int]:
        """
        Returns the set of colours used by vertices adjacent to the given vertex.

        Parameters
        ----------
        v : int
            The vertex for which the adjacent colours are to be determined.

        Returns
        -------
        set[int]
            A set containing the colours used by adjacent vertices. If no adjacent vertices are assigned a colour,
            an empty set is returned.
        """
        adjacent_colours: set = set()
        for u in self.problem.adjacency_list[v]:
            adjacent_colours.add(self.vertex_colours[u])
        return adjacent_colours.difference(*[[0]])

    def _construction_move(self: Solution, move: Move) -> Solution:
        """
        Helper method for Apply - Applies a construction move

        Adds a vertex to the specified colour class in the solution.

        Parameters
        ----------
        self : Solution
            The solution instance that is being modified.
        move : Move
            A `Move` object containing the vertex (`vertex`) and the new colour (`new_colour`) to be assigned.

        Returns
        -------
        Solution
            The updated `Solution` object after the vertex has been added to the specified colour class.
        """
        try:
            if move.new_colour > len(self.colour_classes.keys()):
                self.colour_classes[move.new_colour] = set()
            self.colour_classes[move.new_colour].add(move.vertex)
            self.vertex_colours[move.vertex] = move.new_colour

            # Update conflict matrix
            for neighbour in self.problem.adjacency_list[move.vertex]:
                self.conflicts[neighbour][move.new_colour][0] += 1
                self.conflicts[neighbour][move.new_colour][1].add(move.vertex)


        except Exception as e:
            logging.error(e)
        return self

    def _destruction_move(self: Solution, move: Move) -> Solution:
        """
        Helper method for Apply - Applies a destruction move

        removes a vertex from the specified colour class in the solution.

        Parameters
        ----------
        self : Solution
            The solution instance that is being modified.
        move : Move
            A `Move` object containing the vertex (`vertex`) and the old colour (`old_colour`) to be removed.

        Returns
        -------
        Solution
            The updated `Solution` object after the vertex has been removed from the specified colour class.
        """
        try:
            # Update conflict matrix
            for neighbour in self.problem.adjacency_list[move.vertex]:
                self.conflicts[neighbour][move.old_colour][0] -= 1
                self.conflicts[neighbour][move.old_colour][1].remove(move.vertex)

                count, conflicts_set = self.conflicts[neighbour][move.old_colour]
                if count == 0 and not conflicts_set:
                    del self.conflicts[neighbour][move.old_colour]

            self.vertex_colours[move.vertex] = 0
            self.colour_classes[move.old_colour].remove(move.vertex)

            if len(self.colour_classes[move.old_colour]) == 0:
                self._swap(move.old_colour)



        except ValueError:
            logging.error(f"Error: {move.vertex} not found in {self.colour_classes[move.old_colour]}")
        except KeyError:
            logging.error(f"Error: {move.vertex} not found in Conflicts neighbour: {neighbour}")
        except Exception as e:
            logging.error(e)
        return self

    def _one_exchange(self: Solution, move: Move) -> Solution:
        """
        Helper method for Apply - Applies a 1-exchange move

        Performs a recolouring operation on a vertex by exchanging its colour.

        Parameters
        ----------
        self : Solution
            The solution instance that is being modified.
        move : Move
            A `Move` object containing the vertex (`vertex`), the old colour (`old_colour`),
             and the new colour (`new_colour`).

        Returns
        -------
        Solution
            The updated `Solution` object after the recolouring operation.
        """
        try:
            if move.new_colour > len(self.colour_classes) + 1:
                raise Exception("Can't assign colour larger than #used_colour + 1")

            if move.old_colour == 0:
                raise Exception("Recolour can't assign colour")

            if move.new_colour > len(self.colour_classes.keys()):
                self.colour_classes[move.new_colour] = set()

            # Update conflict matrix
            for neighbour in self.problem.adjacency_list[move.vertex]:
                self.conflicts[neighbour][move.old_colour][0] -= 1
                self.conflicts[neighbour][move.old_colour][1].remove(move.vertex)
                self.conflicts[neighbour][move.new_colour][0] += 1
                self.conflicts[neighbour][move.new_colour][1].add(move.vertex)

                count, conflicts_set = self.conflicts[neighbour][move.old_colour]
                if count == 0 and not conflicts_set:
                    del self.conflicts[neighbour][move.old_colour]

            self.colour_classes[self.vertex_colours[move.vertex]].remove(move.vertex)
            self.vertex_colours[move.vertex] = move.new_colour

            self.colour_classes[move.new_colour].add(move.vertex)

            if len(self.colour_classes[move.old_colour]) == 0:
                self._swap(move.old_colour)

        except Exception as e:
            logging.error(e)
        return self

    def _swap(self, colour: int):
        """
        Swaps the given colour class with the highest non-empty colour class. Used to remove holes in the colours.

        This method performs a swap between the specified colour class (`colour`) and the highest non-empty colour
        class. If the specified colour is already the highest, it is simply removed. The vertex assignments for the
        colour class are updated accordingly. After the operation, the move generator is reset.

        Parameters
        ----------
        colour : int
            The colour to be swapped with the highest non-empty colour class.

        Returns
        -------
        None
            This method modifies the internal state of the solution in-place and does not return a value.
        """
        try:
            assert colour in self.colour_classes.keys()

            highest_colour = len(self.colour_classes.keys())

            if highest_colour == colour:
                for vertex in self.colour_classes[colour]:
                    del self.conflicts[vertex][colour]

                # If the colour is already the highest, just remove it
                del self.colour_classes[colour]
            else:
                # Find the highest non-empty colour
                highest_with_vertex = None
                for i in range(highest_colour, 0, -1):
                    if self.colour_classes[i]:
                        highest_with_vertex = i
                        break

                if highest_with_vertex is not None:

                    # Update conflict
                    for vertex in self.vertex_colours.keys():
                        self.conflicts[vertex][colour] = self.conflicts[vertex][highest_with_vertex]
                        del self.conflicts[vertex][highest_with_vertex]

                        count, conflicts_set = self.conflicts[vertex][colour]
                        if count == 0 and not conflicts_set:
                            del self.conflicts[vertex][colour]

                    # Swap colour classes
                    self.colour_classes[colour] = self.colour_classes.pop(highest_with_vertex)

                    # Update vertex colours
                    for v in self.colour_classes[colour]:
                        self.vertex_colours[v] = colour

        except Exception as e:
            logging.error(e)
        # Reset move_generator after changing colour classes
        self._reset_move_generator()
        return

    def _total_conflicts_in_colour_class(self, colour_class):
        # Sum up all conflicts for the given colour_class across all vertices and divide by 2 to avoid double-counting
        total_conflicts = 0
        for vertex in self.colour_classes.get(colour_class, {}):
            total_conflicts += self.conflicts[vertex][colour_class][0]  # Add conflicts for the given colour_class
        return total_conflicts // 2  # Divide by 2 to avoid double-counting

    def _reset_move_generator(self):
        """
        Resets the move generators for local, construction, and destruction moves.

        This method initializes the iterators for generating random moves without replacement in the solution.
        Returns
        -------
        None
            This method modifies the internal state of the solution in place and does not return a value.
        """
        self.local_move_generator = iter(sample2(self.problem.vertex_count,
                                                       min(len(self.colour_classes.keys()) + 1,
                                                           self.problem.vertex_count)))
        self.construction_move_generator = iter(sample2(self.problem.vertex_count,
                                                              min(len(self.colour_classes.keys()) + 1,
                                                                  self.problem.vertex_count)))
        self.destruction_move_generator = iter(sample(self.problem.vertex_count))

    def check_colouring_constraints(self):
        return check_colouring_constraints(self)


class Construction_neighbourhood(NeighbourhoodInterface):
    def __init__(self, problem: Problem):
        self.problem: Problem = problem

    
    def moves(self: Neighbourhood, solution: Solution) -> Iterable[Move]:
        """
        Generates all possible construction moves for the given solution.

        Parameters
        ----------
        solution : Solution
            The current solution state, which contains the vertex assignments and colour classes.

        Returns
        -------
        Iterable[Move]
            An iterable (set) of valid construction moves. Each move represents the assignment of a new colour
            to an unassigned vertex.
        """
        _moves = set()
        for vertex, c in solution.vertex_colours.items():
            if c == 0:
                # Range from 1 to minimum of # of use colours and # of vertices
                colours = min(len(solution.colour_classes.keys()) + 1, self.problem.vertex_count)
                adding_moves = {Move(move_type=Move_type.CONSTRUCTION_MOVE, vertex=vertex, new_colour=new_colour)
                                for new_colour in range(1, colours + 1)}
                _moves.update(adding_moves)
        return _moves

    
    def random_move(self: Neighbourhood, solution: Solution) -> Optional[Move]:
        """
        Generates a random construction move for the given solution.

        This method randomly selects an unassigned vertex (colour 0) and assigns it a random new colour. The new colour
        is chosen from the available colours, based on the number of used colours and the total number of vertices.

        Parameters
        ----------
        solution : Solution
            The current solution state.

        Returns
        -------
        Optional[Move]
            A random construction move, represented as a `Move` object, or `None` if no valid move can be generated.
            The move consists of a vertex and a new colour assignment.
        """
        try:
            vertex = choice([v for v, c in solution.vertex_colours.items() if c == 0])
            # Range from 1 to minimum of # of use colours and # of vertices
            new_colour = choice(
                [c for c in range(1, min(len(solution.colour_classes.keys()) + 1, self.problem.vertex_count) + 1)])

            if vertex is not None and new_colour is not None:
                return Move(move_type=Move_type.CONSTRUCTION_MOVE, vertex=vertex, new_colour=new_colour)
            return None
        except Exception:
            return None
    
    def random_moves_without_replacement(self: Neighbourhood, solution: Solution) -> Optional[Move]:
        """
        Returns a valid move from the move generator. It continues selecting moves without
        replacement until a valid move is found (i.e., an unassigned vertex is assigned a new colour). If no valid moves
        are available, it returns `None`.

        Parameters
        ----------
        solution : Solution
            The current solution state, which contains the vertex colour assignments and move generator.

        Returns
        -------
        Optional[Move]
            A valid construction move represented as a `Move` object, or `None` if no valid move is found.
        """
        try:
            while True:  # Keep trying until we find a valid move
                vertex, new_colour = next(solution.construction_move_generator)
                if vertex and new_colour and solution.vertex_colours[vertex] == 0:
                    return Move(Move_type.CONSTRUCTION_MOVE, vertex, new_colour)

        except StopIteration:
            # Generator is exhausted, return None
            return None


class Destruction_neighbourhood(NeighbourhoodInterface):
    def __init__(self, problem: Problem):
        self.problem: Problem = problem

    
    def moves(self: Neighbourhood, solution: Solution) -> Iterable[Move]:
        """
        Generates all possible destruction moves for the given solution.

        Parameters
        ----------
        solution : Solution
            The current solution state, which contains the vertex assignments and colour classes.

        Returns
        -------
        Iterable[Move]
            An iterable (set) of valid destruction moves. Each move represents the assignment of a new colour
            to an unassigned vertex.
        """
        _moves = set()
        for vertex, colour in solution.vertex_colours.items():
            if colour != 0:
                _moves.update({Move(Move_type.DESTRUCTION_MOVE, vertex, old_colour=colour, new_colour=0)})
        return _moves

    
    def random_move(self: Neighbourhood, solution: Solution) -> Optional[Move]:
        """
        Generates a random destruction move for the given solution, by randomly selecting a coloured vertex.

        Parameters
        ----------
        solution : Solution
            The current solution state.

        Returns
        -------
        Optional[Move]
            A random construction move, represented as a `Move` object, or `None` if no valid move can be generated.
            The move consists of a vertex and a new colour assignment.
        """
        try:
            vertex_with_colour = [vertex for vertex, colour in solution.vertex_colours.items() if colour != 0]
            vertex = choice(vertex_with_colour)

            if vertex:
                return Move(Move_type.DESTRUCTION_MOVE, vertex, old_colour=solution.vertex_colours[vertex], new_colour=0)
            return None
        except Exception:
            return None

    
    def random_moves_without_replacement(self: Neighbourhood, solution: Solution) -> Optional[Move]:
        """
        Returns a valid move from the move generator. It continues selecting moves without
        replacement until a valid move is found (i.e., an unassigned vertex is assigned a new colour). If no valid moves
        are available, it returns `None`.

        Parameters
        ----------
        solution : Solution
            The current solution state.

        Returns
        -------
        Optional[Move]
            A valid construction move represented as a `Move` object, or `None` if no valid move is found.
        """
        try:
            while True:  # Keep trying until we find a valid move
                vertex = next(solution.destruction_move_generator)
                if vertex is not None:
                    return Move(Move_type.DESTRUCTION_MOVE, vertex, old_colour=solution.vertex_colours[vertex], new_colour=0)
        except StopIteration:
            # Generator is exhausted, return None
            return None


class Local_neighbourhood(NeighbourhoodInterface):
    def __init__(self, problem: Problem):
        self.problem: Problem = problem

    
    def moves(self: Neighbourhood, solution: Solution) -> Iterable[Move]:
        """
        Generates all possible 1-exchange moves for the given solution.

        Parameters
        ----------
        solution : Solution
            The current solution state, which contains the vertex assignments and colour classes.

        Returns
        -------
        Iterable[Move]
            An iterable (set) of valid 1-exchange moves. Each move represents the assignment of a new colour
            to an unassigned vertex.
        """
        solution.has_gen_local_moves = True
        _moves = set()
        for vertex, old_colour in solution.vertex_colours.items():
            # Range from 1 to minimum of # of use colours and # of vertices
            colours = min(len(solution.colour_classes.keys()) + 1, self.problem.vertex_count)
            # Same filtering of moves as in random_moves_without_replacement's if, elif, else stmt
            vertex_moves = {Move(Move_type.ONE_EXCHANGE, vertex, new_colour, old_colour)
                            for new_colour in range(1, colours + 1) if new_colour != old_colour}
            _moves.update(vertex_moves)
        return _moves

    
    def random_move(self: Neighbourhood, solution: Solution) -> Optional[Move]:
        """
        Generates a random construction move for the given solution.

        This method randomly selects a coloured vertex (colour 0) and assigns it a new colour. The new colour
        is chosen from the available colours, based on the number of used colours and the total number of vertices.

        Parameters
        ----------
        solution : Solution
            The current solution state.

        Returns
        -------
        Optional[Move]
            A random construction move, represented as a `Move` object, or `None` if no valid move can be generated.
            The move consists of a vertex and a new colour assignment.
        """
        try:
            vertex = choice([v for v, c in solution.vertex_colours.items() if c != 0])
            old_colour = solution.vertex_colours[vertex]
            # Range from 1 to minimum of # of use colours and # of vertices
            colours = min(len(solution.colour_classes.keys()) + 1, self.problem.vertex_count)
            # Same filtering of moves as in random_moves_without_replacement's if, elif, else stmt
            new_colour = choice([new_colour for new_colour in range(1, colours + 1) if new_colour != old_colour])

            return Move(Move_type.ONE_EXCHANGE, vertex, new_colour, old_colour)
        except Exception:
            return None

    
    def random_moves_without_replacement(self: Neighbourhood, solution: Solution) -> Optional[Move]:
        """
        Returns a valid move from the move generator. It continues selecting moves without
        replacement until a valid move is found (i.e., an unassigned vertex is assigned a new colour). If no valid moves
        are available, it returns `None`.

        Parameters
        ----------
        solution : Solution
            The current solution state, which contains the vertex colour assignments and move generator.

        Returns
        -------
        Optional[Move]
            A valid construction move represented as a `Move` object, or `None` if no valid move is found.
        """
        try:
            while True:  # Keep trying until we find a valid move
                vertex, new_colour = next(solution.local_move_generator)
                old_colour = solution.vertex_colours[vertex]
                move = Move(Move_type.ONE_EXCHANGE, vertex, new_colour, old_colour)
                if new_colour != old_colour:
                    return move

        except StopIteration:
            # Generator is exhausted, return None
            return None


class Move_type(Enum):
    CONSTRUCTION_MOVE = 1
    DESTRUCTION_MOVE = 2
    ONE_EXCHANGE = 3

class Move(MoveInterface):
    def __init__(self, move_type: Move_type, vertex: int, new_colour: Optional[int] = None,
                 old_colour: Optional[int] = None):
        self.move_type = move_type
        self.vertex: int = vertex
        self.new_colour: Optional[int] = new_colour
        self.old_colour: Optional[int] = old_colour
        self._cached_hash = hash((self.move_type, self.vertex, self.old_colour, self.new_colour))

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self._cached_hash == other._cached_hash

    def __hash__(self):
        return self._cached_hash

    def __repr__(self):
        return f"Move(move_type={self.move_type}, vertex={self.vertex}, new_colour={self.new_colour}, old_colour={self.old_colour})"

    def invert(self: Move) -> Move:
        """
        Returns the inverse of the current move.

        This method constructs and returns the logical inverse of the move. For example:
        - A `CONSTRUCTION_MOVE` becomes a `DESTRUCTION_MOVE` with the `old_colour` set to the original `new_colour`.
        - A `DESTRUCTION_MOVE` becomes a `CONSTRUCTION_MOVE` with the `new_colour` set to the original `old_colour`.
        - A `ONE_EXCHANGE` move swaps the `old_colour` and `new_colour`.

        Returns
        -------
        Move
            A new `Move` object representing the inverse of the original move.
        """
        match self.move_type:
            case Move_type.CONSTRUCTION_MOVE:
                return Move(move_type=Move_type.DESTRUCTION_MOVE, vertex=self.vertex, old_colour=self.new_colour)
            case Move_type.DESTRUCTION_MOVE:
                return Move(move_type=Move_type.CONSTRUCTION_MOVE, vertex=self.vertex, new_colour=self.old_colour)
            case Move_type.ONE_EXCHANGE:
                return Move(move_type=Move_type.ONE_EXCHANGE, vertex=self.vertex, new_colour=self.old_colour, old_colour=self.new_colour)
            case _:
                # Can't happen
                pass



def gen_sol_file(path: Path, solution: Solution):
    with open(path, "w") as file:
        for i in range(solution.problem.vertex_count):
            if i == solution.problem.vertex_count - 1:
                file.write(str(solution.vertex_colours[i]))
            else:
                file.write(str(solution.vertex_colours[i]) + "\n")


def check_colouring_constraints(solution: Solution):
    constraints_violated_text = []
    constraints_violated_nodes = []
    num_nodes = len(solution.problem.adjacency_matrix)

    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if solution.problem.adjacency_matrix[u][v] == 1:
                if solution.vertex_colours[u] == solution.vertex_colours[v]:
                    constraints_violated_text.append(
                        f"Constraint violated: Node {u + 1} and Node {v + 1} have the same colour.")
                    constraints_violated_nodes.append((u + 1, v + 1))

    return constraints_violated_text, constraints_violated_nodes

def get_numbers_from_lines(file_path, line1, line2):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if line1 <= len(lines) and line2 <= len(lines):
        num1 = int(lines[line1 - 1].strip())
        num2 = int(lines[line2 - 1].strip())
        return num1, num2
    else:
        raise ValueError(f"File has less than {max(line1, line2)} lines.")



if __name__ == '__main__':

    def main():
        import argparse
        import sys

        parser = argparse.ArgumentParser()
        parser.add_argument('--input-file', type=argparse.FileType('r'), default=sys.stdin)

        args = parser.parse_args()
        filename = args.input_file.name

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")

        # set Plot to True to show plot during run
        metric_log_file_path, logger = add_metric_logger(f"runs/metric_logs/{os.path.splitext(os.path.basename(filename))[0]}.csv", False)
        sol_file_path = Path(f"runs/solution_files/{str(metric_log_file_path).rsplit('/', 1)[-1].rsplit('.', 1)[0]}.sol")

        if filename.endswith(".txt"):
            p = Problem.from_textio(args.input_file)
        elif filename.endswith(".col"):
            p = Problem.from_col(args.input_file)
        else:
            logging.error("Unsupported file format")


        s1: Optional[Solution] = p.heuristic_solution()

        # s1 = tabu_search(problem=p, current_solution=s1, min_tabu_tenure=10, max_tabu_tenure=100, max_iterations=1000000, aspiration_criteria=-10, clean_iter=10000, idle_iterations=1000000, metric_logger=logger)
        # s1 = iterated_local_search(p, s1, 1000000, perturbation_strength=int(p.vertex_count / 100 * 2.0), logger=logger)
        s1 = simulated_annealing(p, s1, 0.5,16,0.1, 1,5,0.02, logger=logger)

        try:
            for colour in range(1, len(s1.colour_classes.keys()) + 1):
                assert colour in s1.colour_classes.keys(), f"Assertion failed on line {inspect.currentframe().f_lineno} for colour {colour}. Colour not found in colour_classes."
                assert len(s1.colour_classes[colour]) != 0, f"Assertion failed on line {inspect.currentframe().f_lineno} for colour_class {colour}. Colour_class is empty."
        except Exception as e:
            logging.error(e)

        print(f"\nSolution s1 Tabu - colours used: {len(s1.colour_classes)}\n")
        # print(f"Vertex_colours: {s1.vertex_colours}\n")

        gen_sol_file(sol_file_path, s1)
        print(run_solution_checker(filename, sol_file_path, 1))
        text, nodes = check_colouring_constraints(s1)
        if text:
            print(text)
            for node_pair in nodes:
                print(f"nodes: {node_pair} has colours {get_numbers_from_lines(sol_file_path, node_pair[0], node_pair[1])}")
        else:
            plot_evaluation(metric_log_file_path)

    main()
    # cProfile.run('main()')