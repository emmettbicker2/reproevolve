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
After running for 1,400 iterations, I got the following program:
![A depiction of a circle packing with a sum of radii=2.627](/images/best_program.png)

The score is 99.7% of what OpenEvolve was able to achieve. The remaining discrepancy is likely based on small adjustments to their algorithm causing it to converge faster, such as sampling better programss as parents more frequently. However, this repository is mainly intended as an educational repository and I am happy with its current level of performance!

# Example Run Command
`poetry run python main.py --initial-program example_initial_program.py --evaluator-program example_evaluator.py`
