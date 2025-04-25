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
import subprocess
import altair as alt
import numpy as np
alt.data_transformers.disable_max_rows()
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, FuncFormatter
from typing import Dict, Set, List, Tuple

def adjacency_matrix(v: int, edge_list: List[Tuple[int, int]]) -> list[list[int]]:
    """Creates an adjacency matrix using plain Python lists."""
    matrix = [[0 for _ in range(v)] for _ in range(v)]
    for vertex1, vertex2 in edge_list:
        matrix[vertex1][vertex2] = 1
        matrix[vertex2][vertex1] = 1
    return matrix


def adjacency_list(v: int, edge_list: List[Tuple[int, int]]) -> Dict[int, Set[int]]:
    """Creates an adjacency list using sets for efficient membership checks."""
    adj_list = {i: set() for i in range(v)}
    for vertex1, vertex2 in edge_list:
        adj_list[vertex1].add(vertex2)
        adj_list[vertex2].add(vertex1)
    return adj_list


def print_matrix(matrix: np.ndarray):
    """Prints an adjacency matrix in a formatted way."""
    if matrix.size == 0:
        print("\nEmpty matrix:")
        return

    col_widths = [max(len(str(cell)) for cell in col) for col in matrix.T]
    print("\nAdjacency matrix:")
    for row in matrix:
        print(" | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row)))


def print_adj_list(adj_list: Dict[int, np.ndarray]):
    """Prints an adjacency list in a formatted way."""
    print("Adjacency list")
    for node, neighbors in sorted(adj_list.items()):
        neighbors_str = ", ".join(map(str, sorted(neighbors)))
        print(f"{node} -> {neighbors_str}")


def plot_graph_from_adj_matrix(solution):
    """Constructs a graph from an adjacency matrix and plots it."""
    G = nx.from_numpy_array(np.array(solution.problem.adjacency_matrix))
    node_colors = np.array([n for n in solution.vertex_colours.values()])

    pos = nx.spring_layout(G)
    edge_colors = ["red" if node_colors[u] == node_colors[v] else "gray" for u, v in G.edges()]

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700,
            font_size=14, font_weight='bold', edge_color=edge_colors, width=2, edge_cmap=plt.cm.Reds)

    plt.title("Graph with Colored Edges")
    plt.show()

def plot_evaluation(csv_file: str | Path):
    csv_file = Path(csv_file)

    df = pd.read_csv(csv_file)
    iteration_data = df.iloc[:, 0]
    cur_evaluation_data = df.iloc[:, 1]
    best_evaluation_data = df.iloc[:, 2]

    plt.figure(figsize=(10, 6))
    plt.plot(iteration_data, best_evaluation_data, label='Best Evaluation', color='blue')
    plt.plot(iteration_data, cur_evaluation_data, label='Current Evaluation', color='orange')

    plt.title('Evaluation vs Iteration')
    plt.ylabel('Evaluation')

    plt.legend()

    if len(best_evaluation_data) > 1:
        max_iteration = max(iteration_data) if not iteration_data.empty else 0
        min_y = min(min(best_evaluation_data), min(cur_evaluation_data)) - 10
        max_y = max(max(best_evaluation_data), max(cur_evaluation_data)) + 10

        # Padding
        plt.xlim(0, max_iteration + 10)
        plt.ylim(min_y, max_y)

        # Use MaxNLocator to limit the number of ticks
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=20))
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=15))

        ticks = plt.gca().get_xticks()
        if len(ticks) > 0 and ticks[0] > 1000:
            # If the first tick is over 1000, scale the x-axis ticks and format them
            formatter = FuncFormatter(lambda x, _: f'{x / 1000:.0f}')  # Scale down by 1000
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.xlabel("Iteration (×1000)")
        else:
            plt.xlabel("Iteration")

    plt.show(block=True)

def plot_evaluation_marimo(csv_file: str | Path):
    csv_file = Path(csv_file)

    df = pd.read_csv(csv_file)
    iteration_data = df.iloc[:, 0]
    cur_evaluation_data = df.iloc[:, 1]
    best_evaluation_data = df.iloc[:, 2]

    plt.figure(figsize=(10, 6))
    plt.plot(iteration_data, best_evaluation_data, label='Best Evaluation', color='blue')
    plt.plot(iteration_data, cur_evaluation_data, label='Current Evaluation', color='orange')

    plt.title('Evaluation vs Iteration')
    plt.ylabel('Evaluation')

    plt.legend()

    if len(best_evaluation_data) > 1:
        max_iteration = max(iteration_data) if not iteration_data.empty else 0
        min_y = min(min(best_evaluation_data), min(cur_evaluation_data)) - 10
        max_y = max(max(best_evaluation_data), max(cur_evaluation_data)) + 10

        # Padding
        plt.xlim(0, max_iteration + 10)
        plt.ylim(min_y, max_y)

        # Use MaxNLocator to limit the number of ticks
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=20))
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=15))

        ticks = plt.gca().get_xticks()
        if len(ticks) > 0 and ticks[0] > 1000:
            # If the first tick is over 1000, scale the x-axis ticks and format them
            formatter = FuncFormatter(lambda x, _: f'{x / 1000:.0f}')  # Scale down by 1000
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.xlabel("Iteration (×1000)")
        else:
            plt.xlabel("Iteration")

    return plt

def create_interactive_plot(csv_path):
    # Check if the file exists
    file_path = Path(csv_path)
    if not file_path.exists():
        print(f"Error: File '{csv_path}' not found.")
        return None

    try:
        # Load the data
        df = pd.read_csv(csv_path)

        # Create a selector for zooming
        brush = alt.selection_interval(
            bind='scales',
            encodings=['x']  # Allow zooming on x-axis
        )

        # Create long-format DataFrame for Altair
        eval_df = df.melt(
            id_vars=['Iteration', 'Time Used'],
            value_vars=['Current evaluation', 'Best evaluation'],
            var_name='metric',
            value_name='value'
        )
        eval_df['type'] = 'evaluation'

        colors_df = df.melt(
            id_vars=['Iteration', 'Time Used'],
            value_vars=['Current colours used', 'Best colours used'],
            var_name='metric',
            value_name='value'
        )
        colors_df['type'] = 'colors'

        # Combine the data
        combined_df = pd.concat([eval_df, colors_df])

        # Create dual-axis plot by iteration
        iteration_plot = create_dual_axis_plot(combined_df, 'Iteration', brush)

        # Create dual-axis plot by time
        time_plot = create_dual_axis_plot(combined_df, 'Time Used', brush)

        return {
            'iteration_plot': iteration_plot,
            'time_plot': time_plot
        }

    except Exception as e:
        print(f"Error processing the file: {str(e)}")
        return None

def create_dual_axis_plot(data, x_field, brush):
    # Create selection for toggling visibility
    selection = alt.selection_point(fields=['metric'], bind='legend')

    # Define color scheme
    color_scale = alt.Scale(
        domain=['Current evaluation', 'Best evaluation', 'Current colours used', 'Best colours used'],
        range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )

    # Define the base chart properties
    base = alt.Chart(data).encode(
        x=alt.X(f'{x_field}:Q', title=x_field)
    ).properties(
        width=800,
        height=500
    ).add_selection(brush, selection)

    # Create the evaluation metrics chart (left y-axis)
    eval_chart = base.mark_line().encode(
        y=alt.Y('value:Q',
                title='Evaluation Score',
                scale=alt.Scale(zero=False)),
        color=alt.Color('metric:N', scale=color_scale, legend=alt.Legend(title="Metrics")),
        strokeDash=alt.condition(
            alt.datum.metric == 'Best evaluation' or alt.datum.metric == 'Best colours used',
            alt.value([5, 5]),  # dashed line for "best" metrics
            alt.value([1, 0])  # solid line for "current" metrics
        ),
        tooltip=['Iteration:Q', 'value:Q', 'metric:N'],
        opacity=alt.condition(selection, alt.value(1), alt.value(0))
    ).transform_filter(
        alt.datum.type == 'evaluation'
    )

    # Create the colors used chart (right y-axis) - now with black title
    colors_chart = base.mark_line().encode(
        y=alt.Y('value:Q',
                title='Number of Colors',
                axis=alt.Axis(titleColor='black')),  # Changed to black
        color=alt.Color('metric:N', scale=color_scale, legend=alt.Legend(title="Metrics")),
        strokeDash=alt.condition(
            alt.datum.metric == 'Best evaluation' or alt.datum.metric == 'Best colours used',
            alt.value([5, 5]),  # dashed line for "best" metrics
            alt.value([1, 0])  # solid line for "current" metrics
        ),
        tooltip=['Iteration:Q', 'value:Q', 'metric:N'],
        opacity=alt.condition(selection, alt.value(1), alt.value(0))
    ).transform_filter(
        alt.datum.type == 'colors'
    )

    # Combine the charts
    combined_chart = alt.layer(
        eval_chart,
        colors_chart
    ).resolve_scale(
        y='independent',
        color='shared'  # Use shared color scale for consistent legend
    ).properties(
        title=f'Optimization Progress by {x_field}'
    )

    return combined_chart

def run_solution_checker(instance_file: str | Path, solution_file: str | Path, problem_type: int = 1):
    verifier_path = Path("verifier_src/coloring-verifier")
    instance_file = Path(instance_file)
    solution_file = Path(solution_file)
    try:
        result = subprocess.run([
            verifier_path, "-i", str(instance_file), "-s", str(solution_file), "-p", str(problem_type)
        ], capture_output=True, text=True, check=True)
        return "Solution Checker Output:\n" + result.stdout
    except subprocess.CalledProcessError as e:
        return "Error running solution checker:\n" + e.stderr