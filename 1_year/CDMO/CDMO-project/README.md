# CDMO-project

Project exam for the course in _"Combinatorial Decision Making and Optimization"_ of the Master's degree in Artificial Intelligence, University of Bologna (2023/2024). 
The aim of this project is to model and solve the **Multiple Courier Planning Problem**—a variation of the Capacitated Vehicle Routing Problem (CVRP), using four different methodologies:

1. **Constraint Programming (CP)**
2. **Propositional Satisfiability (SAT)**
3. **Satisfiability Modulo Theories (SMT)**
4. **Mixed-Integer Linear Programming (MIP)**

Each approach will be evaluated to understand its strengths and limitations in solving this combinatorial optimization challenge.


## Docker Instructions

1. Open the terminal.
2. Navigate to the directory containing this README.
3. To build the Docker image, choose a name and run:

   `docker build . -t <image_name> -f Dockerfile`
4. Launch the Docker image in a bash shell to use it:

   `docker run -it <image_name> /bin/bash`
5. The CLI usage instructions will be displayed in the terminal. You can also refer to them [in the following section](#execution-of-the-main).
6. After all executions, to extract the results, first obtain the container ID with:

   `docker ps -qf "ancestor=<image_name>"`

   Then, extract the folder with:

   `docker cp <container_id>:<folder_path_in_container> <folder_path_in_local>`

   **Note**: When specifying the path in the container, remember that the entire project is located in `/project`.

## Execution of the main

The `main.py` script is the primary entry point for running different combinatorial optimization approaches on specified instances. The script can be executed with various command-line options, allowing for flexible configuration and execution.

### Basic Usage

Running `main.py` without any arguments will solve all instance numbers using all available approaches (`cp`, `sat`, `smt`, `mip`). However, you can customize the execution by specifying particular instances, approaches, or a configuration file. More than one approach can be used to solve one or more instances.

### Command-line Options

- `-h, --help`

  Displays the help message and exits.
- `-i, -instances INSTANCES [INSTANCES ...]`

  Specify one or more instance numbers to execute.
- `-config CONFIG`

  Provide a JSON configuration file to customize execution settings such as folder names or instance numbers.
- `-cp`

  Run the CP (Constraint Programming) approach.
- `-sat`

  Run the SAT (Satisfiability) approach.
- `-smt`

  Run the SMT (Satisfiability Modulo Theories) approach.
- `-mip`

  Run the MIP (Mixed-Integer Programming) approach.
- `-k, -keep_folder`

  Prevent the results folder from being emptied during subsequent executions from the CLI.

### Configuration File

The configuration file is a JSON that can be used to define settings such as folder names and instance numbers. Below is an example of a JSON configuration. All the keys must be present:

```json
{
    "instances_folder": "instances_dzn",
    "smt_models_folder": "SMT/models",
    "results_folder": "res",
    "indent_results": true,
    "first": 1,
    "last": 21,
    "timeout": 300,
    "run_checker": true
}
```

### Result Folder

If an instance or an approach is parsed from the CLI, the results will be stored in a `<results_folder>_temp` folder.

### Example Usage

- `python.exe main.py -cp -i 1`
- `python.exe main.py -k -cp -sat -mip -i 1 2 3`
- `python.exe main.py -k -cp -sat -i 1 2 3 -config config.json`

### SMT Models

When solving using SMT, a model is compiled as a `.smt2` file and stored in the designated folder. Other solvers could then used by directly loading the model.

### File Converter

If presenting with new _.dat_ instances to parse into _.dzn_, `fileConverter.py` can be used by specifing the source and destination folder.

### Authors

• [Davide Crociati](https://github.com/davidecrociati)

• [Davide Sonno](https://github.com/davidesonno)

• [Lorenzo Pellegrino](https://github.com/lollopelle01)
