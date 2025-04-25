from API_Implementation import marimo, marimo as mo

__generated_with = "0.11.21"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# ROAR-NET API Specification examples - Interface""")
    return


@app.cell
def _(mo):
    import pkgutil
    import importlib as importlib
    import importlib.util
    import inspect as inspect
    import sys
    import os


    def get_solvers():
        module = importlib.import_module("API_Implementation.verifier_src.algos")
        members = inspect.getmembers(module, inspect.isfunction)
        return {x[0]: x[1] for x in members}


    def get_models():
        module = importlib.import_module("API_Implementation.verifier_src.problem_models_package")
        members = inspect.getmembers(module, inspect.isclass)

        filtered = []
        member_dict = {}

        for name, cls in members:
            try:
                module_parts = cls.__module__.split(".")
                if module_parts[-2] == "problem_models_package":
                    submodule_name = module_parts[-1]

                    if submodule_name not in member_dict:
                        member_dict[submodule_name] = [
                            None,
                            None,
                        ]

                    if "Problem" in name:
                        member_dict[submodule_name][0] = cls
                    elif "Solution" in name:
                        member_dict[submodule_name][1] = cls

                    filtered.append((name, cls))
            except IndexError:
                pass

        return member_dict


    solvers = get_solvers()
    models = get_models()

    input_fields = mo.ui.array([])
    user_inputs = mo.ui.dictionary({})
    solver_params = mo.ui.array(input_fields, label="Solver Params")


    def update_solver_params(function):
        new_input_fields = []
        new_user_inputs = {}
        signature = inspect.signature(function)

        for param_name, param in signature.parameters.items():
            label = f"{param_name} ({param.annotation if param.annotation != inspect.Parameter.empty else 'No type specified'})"

            if param.default != inspect.Parameter.empty:
                input_field = mo.ui.text(
                    label=label,
                    value=str(param.default),
                    on_change=lambda value, name=param_name: new_user_inputs.update(
                        {name: value}
                    ),
                )
            else:
                input_field = mo.ui.text(
                    label=label,
                    on_change=lambda value, name=param_name: new_user_inputs.update(
                        {name: value}
                    ),
                )

            new_input_fields.append(input_field)

        input_fields = mo.ui.array(new_input_fields)
        user_inputs = mo.ui.dictionary(new_user_inputs)
        return


    def fist(dict: dict):
        return list(dict.keys())[0]


    selected_model = mo.ui.dropdown(
        label="Select Model",
        options=models.keys(),
    )

    selected_solver = mo.ui.dropdown(
        label="Select Solver",
        options=solvers.keys(),
        on_change=lambda value: (
            print(f"Selected value: {value}"),  # Debugging line
            update_solver_params(solvers.get(value)),
        ),
    )
    return (
        fist,
        get_models,
        get_solvers,
        importlib,
        input_fields,
        inspect,
        models,
        os,
        pkgutil,
        selected_model,
        selected_solver,
        solver_params,
        solvers,
        sys,
        update_solver_params,
        user_inputs,
    )


@app.cell
def _(mo, selected_model, selected_solver, solver_params):
    mo.vstack(
        [
            mo.hstack(
                [selected_model, mo.md(f"Chosen model: {selected_model.value}")]
            ),
            mo.hstack(
                [selected_solver, mo.md(f"Chosen solver: {selected_solver
                .value}")]
            ),
            mo.hstack(
                [solver_params, solver_params.value], justify="space-between"
            ),
        ]
    )
    return


@app.cell
def _(selected_model, selected_solver):
    print(selected_model.value)
    print(selected_solver.value)
    return


if __name__ == "__main__":
    app.run()
