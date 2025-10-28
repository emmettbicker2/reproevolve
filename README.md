# ReproEvolve

This repository is a minimal reproduction of OpenEvolve. This project aims to have code that clearly and legibly represents the key theoretical concepts behind OpenEvolve: the MAP-Elites database, island-based evolution, the evaluator, and the general control flow. It took me a long time to understand these theoretical concepts by reading OpenEvolve's code, so I wanted to make a codebase that prioritizes clear abstractions of the theoretical concepts behind OpenEvolve.

OpenEvolve has 9,086 lines of Python code (as of October 27th, 2025), and this repository has 999 lines of Python code. I hope the reduction makes this repository easier to understand.

## MAP-Elites database + Island Based Evolution
The database is in `reproevolve/database.py`. It is a simple 2D grid of programs. Every program object has a list of islands it belongs to. When a program needs to migrate from one program to another, the new island's ID is added to the program's list of islands.

## Evaluator
The logic for evaluating a program's fitness, along with additional metrics, exists in `reproevolve/evaluator.py`. I matched the dimensions that OpenEvolve uses to define the MAP-Elites grid: code complexity and program diversity.

## Prompt Engineering
Every time a new program is generated, several programs from the database are shown to the model for inspiration. All of the functionality for prompt engineering exists in `reproevolve/prompt_builder.py` and `reproevolve/prompt_templates.py`. I matched OpenEvolve's prompt engineering style.

## Main control loop
The main control loop exists in `reproevolve/controller.py`. The controller's `run` method closely matches a high-level understanding of what OpenEvolve does. In pseudocode, the function
- Selects the next island
- Gets a random program from that island
- Generates a child program from this program
- Evaluates the child program
- Updates the database and potentially migrates this new program to a new island

# Results
After running the algorithm for 2,500 iterations, the final program achieves a fitness of ~2.629

<p align="center">
  <img width="768" height="734" alt="Circle Packing with 2.629" src="https://github.com/user-attachments/assets/6c00f380-9aea-4735-9a3d-9295a68ccd97" />
</p>

This is a visualization of the best program's fitness over time:
<p align="center">
<img width="554" height="340" alt="Graph of performance over time" src="https://github.com/user-attachments/assets/1f0061a6-2700-4423-b2e3-de6114deed2e" />
</p>


The score is 99.8% of what OpenEvolve was able to achieve (2.634). OpenEvolve employs additional strategies to speed up convergence in its circle packing example. A few examples are: limiting the database to the 60 highest-performing programs, deliberately sampling parents with high fitness, and a method for island migration that prioritizes migrating programs with high fitness. However, this repository is intended for educational purposes, so I find its current level of performance perfectly acceptable.

# Example Run Command
`poetry run python main.py --initial-program example_initial_program.py --evaluator-program example_evaluator.py`
