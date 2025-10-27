import os

from reproevolve.database import Program
from reproevolve.evaluator import EvalReturn

BEST_PROGRAM_OUTPUT_PATH = "best_program.py"
LOG_PATH = "log.txt"


def reset_log():
    # Clear log
    if os.path.isfile(LOG_PATH):
        os.remove(LOG_PATH)


def checkpoint(
    model: str, evaluate_return: EvalReturn, island: int, best_program: Program
):
    with open(LOG_PATH, "a") as log:
        log.write(f"model: {model} island: {island} {str(evaluate_return)}\n")

    with open(BEST_PROGRAM_OUTPUT_PATH, "w") as f:
        f.write(best_program.code)
