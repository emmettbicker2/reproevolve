# ReproEvolve

This repository is a minimal reproduction of OpenEvolve. The aim of this repository is for the code to clearly and legibly represent the key theoretical concepts behind OpenEvolve: the MAP-Elites database, island-based evolution, the evaluator, and the general control flow. I have a lot of respect for the creators and maintainers of OpenEvolve, but it took me a long time to understand these theoretical concepts from reading the code alone, so I wanted to make a codebase that prioritizes clear abstractions of the theoretical concepts behind OpenEvolve.

OpenEvolve is 9,086 lines of python code (as of October 27th, 2025), while this repository is 999 lines of python code. I hope the reduction makes this repository easily understandable!

## MAP-Elites database + Island Based Evolution
The database is in `reproevolve/database.py`. It is a simple 2D grid of programs. Every program in this database has a list of islands that it belongs to. When a program needs to migrate from one program to another, the new island's id is added to the program's list of islands.

## Evaluator
The logic for evaluating a program's fitness, along with additional metrics, exists in `reproevolve/evaluator.py`. I matched OpenEvolve's two features of interest: code complexity and program diversity.

## Prompt Engineering
Every time a new program is generated, several programs from the database are shown to the model for inspiration. All of the functionality for prompt engineering exists in `reproevolve/prompt_builder.py` and `reproevolve/prompt_templates.py`. I matched OpenEvolve's prompt engineering style.

## Main control loop
The main control loop exists in `reproevolve/controller.py`. The controller's `run` method closely matches a high level understanding of what OpenEvolve does. In psuedocode, the function
- Gets the next island
- Gets a random program from that island
- Generates a child program from this program
- Evaluate the child program
- Update the database and potentially migrate this new program to a new island

# Results
After running the algorithm for 2,500 iterations, the final program achieves a fitness of ~2.629

<p align="center">
  <img width="768" height="734" alt="Circle Packing with 2.629" src="https://github.com/user-attachments/assets/6c00f380-9aea-4735-9a3d-9295a68ccd97" />
</p>

This is a visualization of the best program's fitness over time:
<p align="center">
<img width="554" height="340" alt="Graph of performance over time" src="https://github.com/user-attachments/assets/1f0061a6-2700-4423-b2e3-de6114deed2e" />
</p>



The score is 99.8% of what OpenEvolve was able to achieve (2.634). There are a few things OpenEvolve does to speed up convergence, such as having a maximum of 60 programs in the database at the same time, deliberately sampling higher-performing parents, and different methods for island migration methods. However, this repository is mainly intended for educational purposes, so  I am happy with its current level of performance!

# Example Run Command
`poetry run python main.py --initial-program example_initial_program.py --evaluator-program example_evaluator.py`
