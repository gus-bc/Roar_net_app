

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from problem_models import graph_colouring
    return graph_colouring, mo


@app.cell
def _():
    import inspect


    def show_method(module, class_name=None, func_name=""):
        if class_name:
            cls = getattr(module, class_name, None)
            if cls:
                if func_name:
                    if func_name == "__init__":
                        # Get class signature
                        class_signature = (
                            f"class {class_name}{inspect.signature(cls)}"
                        )

                        # Get all dunder methods
                        dunder_methods = {
                            name: method
                            for name, method in vars(cls).items()
                            if callable(method)
                            and name.startswith("__")
                            and name.endswith("__")
                        }

                        # Get source code for dunder methods
                        dunder_sources = []
                        for name, method in dunder_methods.items():
                            try:
                                dunder_sources.append(inspect.getsource(method))
                            except TypeError:
                                dunder_sources.append(
                                    f"# Source not available for {name}"
                                )

                        return (
                            f"```python\n{class_signature}\n"
                            + "\n".join(dunder_sources)
                            + "\n```\n"
                        )
                    else:
                        func = getattr(cls, func_name, None)
                else:
                    # Show class definition with all methods
                    class_source = inspect.getsource(cls)
                    return f"```python\n{class_source}\n```\n"
            else:
                return f"Class '{class_name}' not found in module."
        else:
            func = getattr(module, func_name, None)

        return (
            f"```python\n{inspect.getsource(func)}\n```\n"
            if func
            else f"Method or function '{func_name}' not found."
        )
    return inspect, show_method


@app.cell
def _(mo):
    mo.md(
        r"""
        # ROAR-NET API Specification examples 


        The following is an example of how the [ROAR-NET API Specification]() can be used to model the Graph-Colouring problem. 


        ## **Usecase: Graph colouring**

        The graph colouring problem can be formally defined as follows:

        Given an undirected graph \( G = (V,E) \), where \( V \) is the set of vertices and \( E \) is the set of edges.

        A **proper $k$-colouring** of the graph \( G \) is a mapping \( C: V \rightarrow K \), where $K = \{1,2,\dots,k\}$,
        such that for every edge \( (u, v) \in E \), the condition $c(u) \neq c(v)$ holds.

        A **color class** corresponding to a color \( i \in K \) is the set of all vertices assigned the same color. Formally, $K_i = \{ v \in V \;|\; c(v) = i \}.$ The set of **uncolored vertices** is defined as: $V' = V  \setminus \bigcup_{i \in K} K_i.$  

        The **conflict set** \( E' \) is the set of all edges where the vertices connected by the edge have the same color. This can be defined as:   $E' = \bigcup_{i \in K} \{(u,v) \in E \;|\; u,v \in K_i\}$ or equivalently, $E' = \{(u,v) \in E \;|\; c(u) = c(v)\}.$

        A lower and upper bound for a graph $G$ can be set as: $\omega(G) < |K| < \Delta(G)+1$


        ## **Strategy**

        This model uses a k-variable complete coloring strategy, where the number of colors used is controlled during the optimization process. It employs the same evaluation function: $f(s) = - \sum_{i\in K}^{} |K_i|^2 + (|E'| + |V'|)$ as described by Johnson et al. (1991) in their study on simulated annealing for graph coloring and number partitioning. They demonstrated that a penalty function approach was not the best performer when using simulated annealing. However, it was chosen to test the easy of using an evaluation function instead of an objective function, and to see if other heuristics performance better that the results presented by Johnson et al. (1991).

        <br/>
        <br/>
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""

            ### **Implementation: API-Operations**

            The API is implemented as a pyhton package with an Opertaions module. This module implements the types and abstract classes to work as interfaces to be implemented in problem-models:
             """
            ),
            mo.accordion(
                {
                    "View code:": mo.md(
                        r"""
                            ```python

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


                            class MoveInterface(ABC):
                                @abstractmethod
                                def invert(self: Move) -> Move:
                                    raise NotImplementedError

                            Solution = TypeVar('Solution', bound=SolutionInterface)
                            Move = TypeVar('Move', bound=MoveInterface)
                            Neighbourhood = TypeVar('Neighbourhood', bound=NeighbourhoodInterface)
                            Problem = TypeVar('Problem', bound=ProblemInterface)

                            ```

                            """
                    )
                }
            ),
            mo.md(
                r"""
                All operations are implemented as function calls, to have the same function signatures as the specification:
                """
            ),
            mo.accordion(
                {
                    "View code:": mo.md(
                        r"""
                        ```python
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

                        def get_num_colours(solution: Solution) -> int:
                         return solution.get_num_colours()
                        ```
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### **Implementation: Graph colouring problem - Problem**

        The problem is implemented as a class that inherits from the ProblemInterface. The class should implement all the data necessary to represent a particular instance of the Graph Colouring problem. The edges between connected vertecies are reprecented as both an adjecency matrix and adjecency list. The number of vertecies are also saved for easy refrence:
        """
    )
    return


@app.cell
def _(graph_colouring, mo, show_method):
    mo.md(
        f"{
            show_method(
                graph_colouring,
                'Problem',
                '__init__',
            )
        }"
    )
    return


@app.cell
def _(mo):
    mo.md("""The **Problem class** also implements the **local_neighbourhood**, **construction_neighbourhood**, **destruction_neighbourhood** methods, which returns thier respective neighbourhood object by calling their constuctor:""")
    return


@app.cell
def _(mo):
    mo.md(r"""To get a **Solution** instance the **Problem** class implements the following methods:""")
    return


@app.cell
def _(graph_colouring, mo, show_method):
    mo.accordion(
        {
            "View methods: ": mo.md(
                f"{
                    show_method(
                        graph_colouring,
                        'Problem',
                        'empty_solution',
                    )
                }"
                f"{
                    show_method(
                        graph_colouring,
                        'Problem',
                        'random_solution',
                    )
                }"
                f"{
                    show_method(
                        graph_colouring,
                        'Problem',
                        'heuristic_solution',
                    )
                }"
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### **Implementation: Graph colouring problem - Neighborhood**

        The three neighbourhoods, local-, construction- and destruction-neighborhood, are implemented as seperate classes all inheriting the NeighbourhoodInterface class. They all implement their own version of the three methods, *moves*, *random_move* and *random_moves_without_replacement*.:
        """
    )
    return


@app.cell
def _(graph_colouring, mo, show_method):
    mo.accordion(
        {
            "View Construction_neighbourhood: ": mo.md(
                f"{
                    show_method(
                        graph_colouring,
                        'Construction_neighbourhood',
                    )
                }"
            ),
            "View Destruction_neighbourhood: ": mo.md(
                f"{
                    show_method(
                        graph_colouring,
                        'Destruction_neighbourhood',
                    )
                }"
            ),
            "View Local_neighbourhood: ": mo.md(
                f"{
                    show_method(
                        graph_colouring,
                        'Local_neighbourhood',
                    )
                }"
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### **Implementation: Graph colouring problem - Move**

        A move is represented by a move type (Enum), an integer (representing the vertex), and optionally two integers representing the color of the vertex before and after the move. The move class implements the MoveInterface class and has the following methods: *Invert*, which returns the inverse of the move.
        """
    )
    return


@app.cell
def _(graph_colouring, mo, show_method):
    mo.accordion(
        {
            "View classes: ": mo.md(
                f"{
                    show_method(
                        graph_colouring,
                        'Move_type',
                    )
                }"
                f"{
                    show_method(
                        graph_colouring,
                        'Move',
                    )
                }"
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### **Implementation: Graph colouring problem - Solution**
        The Solution class implements the SolutionInterface and implements all data necessary to represent a solution, which is the colors of the vertices. To make some operations easier both a dictionary with vertex as key and colour as value and a dictionary of colour as key and the colour class (a set) as value, is implemented. The solution also contains the problem instance and three move generators, one for each type of neighbourghood.
        """
    )
    return


@app.cell
def _(graph_colouring, mo, show_method):
    mo.accordion(
        {
            "View code: ": mo.md(
                f"{
                    show_method(
                        graph_colouring,
                        'Solution',
                        '__init__',
                    )
                }"
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""The Solution class implements the following methods:""")
    return


@app.cell
def _(graph_colouring, mo, show_method):
    mo.accordion(
        {
            "**objective_value**: ": mo.vstack(
                [
                    mo.md(
                        "As describe in the intro the model uses an evaluation function and not the objective function. However, since evaluation functions isn't supported it is implemented as the objective function. The objective value is calculated as the sum of the square of the size of each colour class, plus the product of number of vertices and the number of conflicts in each colour class."
                    ),
                    mo.md(
                        f"{
                            show_method(
                                graph_colouring,
                                'Solution',
                                'objective_value',
                            )
                        }"
                    ),
                ]
            ),
            "**objective_value_increment**: ": mo.vstack(
                [
                    mo.md(
                        f"{
                            show_method(
                                graph_colouring,
                                'Solution',
                                'objective_value_increment',
                            )
                        }"
                    )
                ]
            ),
            "**lower_bound**: ": mo.vstack(
                [
                    mo.md(
                        f"{
                            show_method(
                                graph_colouring,
                                'Solution',
                                'lower_bound',
                            )
                        }"
                    )
                ]
            ),
            "**lower_bound_increment**: ": mo.vstack(
                [
                    mo.md(
                        f"{
                            show_method(
                                graph_colouring,
                                'Solution',
                                'lower_bound_increment',
                            )
                        }"
                    )
                ]
            ),
            "**apply**: ": mo.vstack(
                [
                    mo.md(
                        f"{
                            show_method(
                                graph_colouring,
                                'Solution',
                                'apply',
                            )
                        }"
                    )
                ]
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""There is also implemented the following helper methods:""")
    return


@app.cell
def _(graph_colouring, mo, show_method):
    mo.accordion(
        {
            "**get_num_colours**: ": mo.vstack(
                [
                    mo.md(
                        f"{
                            show_method(
                                graph_colouring,
                                'Solution',
                                'get_num_colours',
                            )
                        }"
                    )
                ]
            ),
            "**adjacent_colours**: ": mo.vstack(
                [
                    mo.md(
                        f"{
                            show_method(
                                graph_colouring,
                                'Solution',
                                'adjacent_colours',
                            )
                        }"
                    )
                ]
            ),
            "**_construction_move**: ": mo.vstack(
                [
                    mo.md(
                        f"{
                            show_method(
                                graph_colouring,
                                'Solution',
                                '_construction_move',
                            )
                        }"
                    )
                ]
            ),
            "**_destruction_move**: ": mo.vstack(
                [
                    mo.md(
                        f"{
                            show_method(
                                graph_colouring,
                                'Solution',
                                '_destruction_move',
                            )
                        }"
                    )
                ]
            ),
            "**_one_exchange**: ": mo.vstack(
                [
                    mo.md(
                        f"{
                            show_method(
                                graph_colouring,
                                'Solution',
                                '_one_exchange',
                            )
                        }"
                    )
                ]
            ),
            "**_swap**: ": mo.vstack(
                [
                    mo.md(
                        f"{
                            show_method(
                                graph_colouring,
                                'Solution',
                                '_swap',
                            )
                        }"
                    )
                ]
            ),
            "**_reset_move_generator**: ": mo.vstack(
                [
                    mo.md(
                        f"{
                            show_method(
                                graph_colouring,
                                'Solution',
                                '_reset_move_generator',
                            )
                        }"
                    )
                ]
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <br/>
        <br/>
        <br/>
        # Roar-NET API-Interface
        """
    )
    return


@app.cell
def _():
    from roar_net_api.data import graph
    from roar_net_api.api.utils import create_interactive_plot
    from roar_net_api.api.custom_logger import add_metric_logger
    from roar_net_api.api.utils import plot_evaluation_marimo
    from roar_net_api import algos
    import problem_models

    import importlib
    from pathlib import Path
    import logging
    import os
    import types
    import pkgutil


    def get_sub_moduels(module):
        return [
            name
            for _, name, _ in pkgutil.iter_modules(
                [os.path.dirname(module.__file__)]
            )
        ]


    def setup():
        problem_files = sorted(
            [
                os.path.relpath(
                    os.path.join(root, file), os.path.dirname(graph.__file__)
                )
                for root, _, files in os.walk(os.path.dirname(graph.__file__))
                for file in files
            ]
        )
        model_loaders = get_sub_moduels(problem_models)
        solver_functions = get_sub_moduels(algos)
        return problem_files, model_loaders, solver_functions
    return (
        Path,
        add_metric_logger,
        create_interactive_plot,
        graph,
        importlib,
        logging,
        os,
        plot_evaluation_marimo,
        setup,
    )


@app.cell
def _(importlib, logging, mo, setup):
    def on_solver_change(solver):
        if solver:
            # Load the selected module
            selected_solver_module = importlib.import_module(
                f"roar_net_api.algos.{solver}"
            )


    def on_problem_change(problem):
        if problem:
            # Load the selected module
            selected_problem_module = importlib.import_module(
                f"problem_models.{problem}"
            )


    # Setup phase
    problem_files, model_loaders, solver_functions = setup()
    selected_solver_module = None
    selected_problem_module = None
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # UI Elements
    instance_problem_path = mo.ui.dropdown(
        problem_files,
        label="Choose problem instance",
    )

    solver = mo.ui.dropdown(
        solver_functions,
        label="Choose Solver",
        on_change=on_solver_change,
    )

    problem = mo.ui.dropdown(
        model_loaders,
        label="Choose problem model",
        on_change=on_problem_change,
    )

    solver_init = mo.ui.dropdown(
        ["Random", "Heuristic", "Empty"],
        label="Choose solution initiation method",
    )
    return (
        instance_problem_path,
        problem,
        selected_problem_module,
        selected_solver_module,
        solver,
        solver_init,
    )


@app.cell
def _(
    Path,
    add_metric_logger,
    create_interactive_plot,
    graph,
    importlib,
    inspect,
    instance_problem_path,
    mo,
    os,
    problem,
    selected_problem_module,
    selected_solver_module,
    solver,
    solver_init,
):
    class Runner:
        def __init__(self):
            self.can_run = False
            self.instance_problem_path = instance_problem_path
            self.solver = solver
            self.problem = problem
            self.solver_init = solver_init
            self.plots = None

            self.solver_options = (
                self.get_params(
                    selected_problem_module,
                    selected_solver_module,
                )
                if self.solver.value and self.solver_init.value
                else []
            )

        def get_params(
            self, selected_problem_module=None, selected_solver_module=None
        ):
            if selected_problem_module is None:
                selected_problem_module = importlib.import_module(
                    f"problem_models.{self.problem.value}"
                )
            if selected_solver_module is None:
                selected_solver_module = importlib.import_module(
                    f"roar_net_api.algos.{self.solver.value}"
                )
            method = selected_solver_module.__dict__[self.solver.value]
            print(f"method: {method}")
            # Get signature and docstring
            sig = inspect.signature(method)
            docstring = inspect.getdoc(method) or ""

            # Basic param docstring parsing
            param_docs = {}
            for line in docstring.splitlines():
                line = line.strip()
                for name, param in sig.parameters.items():
                    if line.startswith(":param " + name):
                        param_docs[param] = line[7:]

            # Build UI inputs
            params = []

            for name, param in sig.parameters.items():
                if name in ("self", "cls"):
                    continue

                help_text = param_docs.get(param, name)

                if param.annotation == int:
                    params.append(
                        mo.ui.number(label=help_text, start=-1e6, stop=1e6, value=0)
                    )
                elif param.annotation == float:
                    params.append(
                        mo.ui.number(
                            label=help_text, start=-1e6, stop=1e6, value=0.0
                        )
                    )
                elif param.annotation == bool:
                    params.append(mo.ui.checkbox(label=help_text))
            print(f"params: {params}")
            return params

        def run(self, selected_problem_module, selected_solver_module):
            if (
                self.problem.value is None
                or self.instance_problem_path.value is None
                or self.solver_init.value is None
                or self.solver.value is None
            ):
                raise ValueError("All parameters must be set before running.")
            else:
                with mo.status.spinner(title="Loading...") as _spinner:
                    if selected_problem_module is None:
                        selected_problem_module = importlib.import_module(
                            f"problem_models.{self.problem.value}"
                        )
                    if selected_solver_module is None:
                        selected_solver_module = importlib.import_module(
                            f"roar_net_api.algos.{self.solver.value}"
                        )

                    problem = None

                    _spinner.update("Running")

                    # set Plot to True to show plot during run
                    metric_log_file_path, metric_logger = add_metric_logger(
                        f"runs/metric_logs/{os.path.splitext(os.path.basename(str(self.instance_problem_path.value)))[0]}.csv",
                        False,
                    )
                    _sol_file_path = Path(
                        f"runs/solution_files/{str(metric_log_file_path).rsplit('/', 1)[-1].rsplit('.', 1)[0]}.sol"
                    )

                    with open(
                        (
                            Path(
                                Path(graph.__file__).parent,
                                self.instance_problem_path.value,
                            )
                        ),
                        mode="r",
                        encoding="utf-8",
                    ) as file:
                        if self.instance_problem_path.value.endswith(".txt"):
                            problem = selected_problem_module.Problem.from_textio(
                                file
                            )
                        elif self.instance_problem_path.value.endswith(".col"):
                            problem = selected_problem_module.Problem.from_col(file)
                        else:
                            print("Unsupported file format")

                    if problem is not None:
                        solution = getattr(
                            selected_problem_module.Problem,
                            (self.solver_init.value.lower() + "_solution"),
                        )(problem)

                        solution = getattr(
                            selected_solver_module, self.solver.value
                        )(
                            problem,
                            solution,
                            *[p.value for p in self.solver_options],
                            metric_logger,
                        )

                        _spinner.update("Done")

                        getattr(selected_problem_module, "gen_sol_file")(
                            _sol_file_path, solution
                        )

                        result = getattr(
                            selected_problem_module, "run_solution_checker"
                        )(self.instance_problem_path.value, _sol_file_path, 1)
                        text, nodes = getattr(
                            selected_problem_module, "check_colouring_constraints"
                        )(solution)
                        if text:
                            print(text)
                            for node_pair in nodes:
                                print(
                                    f"nodes: {node_pair} has colours {getattr(selected_problem_module, 'get_numbers_from_lines')(_sol_file_path, node_pair[0], node_pair[1])}"
                                )
                        else:
                            self.plots = create_interactive_plot(
                                metric_log_file_path
                            )


    runner = Runner()
    return Runner, runner


@app.cell
def _(Runner):
    def can_run(runner: Runner) -> bool:
        if (
            runner.problem.value is not None
            and runner.instance_problem_path.value is not None
            and runner.solver_init.value is not None
            and runner.solver.value is not None
            and all(p.value != None for p in runner.solver_options)
        ):
            return True
        else:
            return False
    return (can_run,)


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(
        label="Run solver",
        kind="danger",
        tooltip="Run the selected solver",
    )
    return (run_btn,)


@app.cell
def _(mo, run_btn, runner, selected_problem_module, selected_solver_module):
    mo.stop(not run_btn.value)
    runner.run(selected_problem_module, selected_solver_module)
    return


@app.cell
def _(can_run, mo, run_btn, runner):
    layout = mo.callout(
        kind="neutral",
        value=mo.vstack(
            [
                runner.problem,
                runner.instance_problem_path
                if runner.problem.value is not None
                else mo.md(""),
                runner.solver
                if runner.instance_problem_path.value is not None
                else mo.md(""),
                runner.solver_init
                if runner.solver.value is not None
                else mo.md(""),
                mo.md(f"**Solver options for**: {runner.solver.value}")
                if runner.solver_init.value
                else mo.md(""),
                *runner.solver_options,
                run_btn if can_run(runner) else mo.md(""),
                runner.plots["iteration_plot"]
                if runner.plots is not None
                else mo.md(""),
                runner.plots["time_plot"]
                if runner.plots is not None
                else mo.md(""),
            ],
            align="center",
        ),
    )
    return (layout,)


@app.cell
def _(layout):
    layout
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <br/>
        ##Previouse Runs
        """
    )
    return


@app.cell
def _(mo, os):
    prev_runs_csv_files = [
        f for f in os.listdir("runs/metric_logs/") if f.endswith(".csv")
    ]
    chosen_run = mo.ui.dropdown(
        prev_runs_csv_files,
        label="Choose problem instance",
    )

    chosen_run
    return (chosen_run,)


@app.cell
def _(Path, chosen_run, create_interactive_plot):
    views = (
        create_interactive_plot(Path(f"runs/metric_logs/{chosen_run.value}"))
        if chosen_run.value
        else None
    )
    return (views,)


@app.cell
def _(mo, views):
    views["iteration_plot"] if views else mo.md("")
    return


@app.cell
def _(mo, views):
    views["time_plot"] if views else mo.md("")
    return


@app.cell
def _(Path, chosen_run, mo, plot_evaluation_marimo):
    plot = (
        plot_evaluation_marimo(Path(f"runs/metric_logs/{chosen_run.value}"))
        if chosen_run.value
        else None
    )

    plot.gca() if plot else mo.md("")
    return


if __name__ == "__main__":
    app.run()
