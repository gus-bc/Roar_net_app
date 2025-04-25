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
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

ALGO_METRIC = 25
logging.addLevelName(ALGO_METRIC, "ALGO_METRIC")



class MetricLogger(logging.Logger):
    def __init__(self, name: str):
        super().__init__(name)
        self._plot = False
        self._is_marimo = False
        self._last_plot_time = 0

        # To store the metric data for graphing purposes
        self.iteration_data = []
        self.cur_evaluation_data = []
        self.best_evaluation_data = []
        self.time_used_data = []
        self.current_colours_used_data = []
        self.best_colours_used_data = []

    def log_algo_metric(self, iteration, cur_evaluation, best_evaluation, time_used, current_colours_used, best_colours_used):
        """
        Logs the algorithm's performance metrics at each iteration.

        Parameters:
        iteration (int): The current iteration number of the algorithm.
        cur_evaluation (float): The evaluation metric value of the algorithm at the current iteration.
        best_evaluation (float): The best evaluation metric value found so far by the algorithm.
        time_used (float): The amount of time (in seconds) used during the current iteration.

        Returns:
        None: This method does not return any value; it is used to log or print the metrics.
        """
        if self.isEnabledFor(ALGO_METRIC):

            # Log the message
            message = f"{iteration},{cur_evaluation},{best_evaluation},{time_used},{current_colours_used},{best_colours_used}"
            self._log(ALGO_METRIC, message, ())

            if _plot and isinstance(iteration, (int, float)):
                self.iteration_data.append(iteration)
                self.cur_evaluation_data.append(cur_evaluation)
                self.best_evaluation_data.append(best_evaluation)
                self.time_used_data.append(time_used)
                self.current_colours_used_data.append(current_colours_used)
                self.best_colours_used_data.append(best_colours_used)

                self.update_graph()

    def update_graph(self):
        global _last_plot_time
        current_time = time.time()

        # Only update the plot if at least 0.5 second has passed
        if current_time - _last_plot_time >= 0.5:
            _last_plot_time = current_time
            plt.clf()

            # Create a single subplot for both curves
            plt.plot(self.iteration_data, self.best_evaluation_data, label='Best Evaluation', color='blue')
            plt.plot(self.iteration_data, self.cur_evaluation_data, label='Current Evaluation', color='orange')

            plt.title('Evaluation vs Iteration')
            plt.xlabel('Iteration')
            plt.ylabel('Evaluation')
            plt.legend()

            if len(self.best_evaluation_data) > 1:
                max_iteration = max(self.iteration_data) if self.iteration_data else 0
                min_y = min(min(self.best_evaluation_data), min(self.cur_evaluation_data)) - 10
                max_y = max(max(self.best_evaluation_data), max(self.cur_evaluation_data)) + 10

                plt.xlim(0, max_iteration + 10)  # Add some padding to the max iteration for clarity
                plt.ylim(min_y, max_y)  # Add padding to y-axis

                # Use MaxNLocator to limit the number of ticks
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=20))  # 20 ticks for x-axis
                plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=15))  # 15 ticks for y-axis

            plt.draw()
            plt.pause(0.05)


def add_metric_logger(file_name: str, plot_during: bool=False, is_marimo: bool=False) -> tuple[Path, MetricLogger]:
    """Creates singleton logger if it doesn't exist yet."""
    if not file_name:
        raise Exception("No provided log filename.")

    global _plot
    _plot = plot_during
    global _is_marimo
    _is_marimo = is_marimo

    _logger = MetricLogger("metric_logger",)

    file_parts = file_name.split(".")
    file_parts[-2] = file_parts[-2] + "_1"
    file_name = ".".join(file_parts)
    path_without_file = "/".join(file_name.split("/")[0:-1])
    while file_name.split("/")[-1] in [file.name for file in Path(path_without_file).iterdir()]:
        file_parts = file_name.split("_")
        base_name = "_".join(file_parts[:-1])
        num_part = int(file_parts[-1].split(".")[0])
        file_name = base_name + "_" + str(num_part + 1) + "." + ".".join(file_parts[-1].split(".")[1:])

    handler = logging.FileHandler(file_name, mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))

    _logger.addHandler(handler)
    _logger.setLevel(ALGO_METRIC)

    _logger.log_algo_metric("Iteration", "Current evaluation", "Best evaluation", "Time Used", "Current colours used", "Best colours used")

    # Initialize the plot
    plt.ion()  # Turn on interactive mode for real-time plotting

    return Path(file_name), _logger
