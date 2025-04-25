def keep_first_half_of_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    half = len(lines) // 2
    with open(file_path, 'w') as file:
        file.writelines(lines[:half])


def delete_x_line_of_file(file_path, x):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if len(lines) < x:
        return
    with open(file_path, 'w') as file:
        file.writelines(lines[:-x])


if __name__ == "__main__":
    delete_x_line_of_file("runs/metric_logs/myciel3_13.csv", 100)