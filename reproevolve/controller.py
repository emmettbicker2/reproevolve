import asyncio
from pydantic import BaseModel
from reproevolve.database import Grid, Metrics, Program
from reproevolve.evaluator import EvalReturn, Evaluator
import re

from reproevolve.logger import checkpoint, reset_log
from reproevolve.model import Model
from reproevolve.prompt_builder import PromptBuilder

INVALID_FITNESS = 0.0
INVALID_PARENT_INDEX = -1


class Config(BaseModel):
    primary_model: str
    secondary_model: str
    secondary_model_usage_rate: float

    grid_size: int
    num_islands: int
    iterations: int
    system_prompt: str
    initial_program_path: str
    evaluator_program_path: str
    timeout_s: int = 60

    migration_rate: float


class Controller:
    """This is our main controller"""

    grid: Grid
    evaluator: Evaluator
    model: Model
    iterations: int

    def __init__(self, config: Config):
        primary_model = config.primary_model
        secondary_model = config.secondary_model
        secondary_model_usage_rate = config.secondary_model_usage_rate

        system_prompt = config.system_prompt
        with open(config.initial_program_path) as f:
            initial_program_code = f.read()

        self.grid = Grid(config.grid_size, config.num_islands, config.migration_rate)
        self.evaluator = Evaluator(config.evaluator_program_path, timeout_s=60)
        self.model = Model(
            primary_model=primary_model,
            secondary_model=secondary_model,
            secondary_model_usage_rate=secondary_model_usage_rate,
            system_prompt=system_prompt,
        )
        self.prompt_builder = PromptBuilder(grid=self.grid)

        self.iterations = config.iterations
        self.best_fitness = INVALID_FITNESS

        # Add the first program to the database
        evaluate_return = asyncio.run(
            self.evaluator.run_evaluation(initial_program_code)
        )
        assert evaluate_return is not None, "Exception on evaluating initial program"
        self.update_database(
            new_program_code=initial_program_code,
            parent_idx=INVALID_PARENT_INDEX,
            islands=self.grid.all_islands,
            evaluate_return=evaluate_return,
        )

        # Clear log
        reset_log()

    def run(
        self,
    ) -> None:
        for i in range(self.iterations):
            island = self.grid.get_next_island()
            model = self.model.choose_model()
            print(f"\nGenerating iteration {i}: selected island {island}")

            old_program = self.grid.random_program(island)
            user_prompt = self.prompt_builder.get_user_prompt(
                program=old_program,
                island=island,
            )

            model_response = self.model.generate_edit(model, user_prompt)
            if model_response is None:
                continue

            new_program_code = apply_diff(
                original_code=old_program.code, diff_text=model_response
            )
            evaluate_return = asyncio.run(
                self.evaluator.run_evaluation(new_program_code)
            )

            # Update database and potentially migrate the program
            self.update_database(
                new_program_code=new_program_code,
                parent_idx=self.grid.get_index_from_program(old_program),
                islands=[island],
                evaluate_return=evaluate_return,
            )
            checkpoint(
                model=model,
                evaluate_return=evaluate_return,
                island=island,
                best_program=self.grid.best_program(islands=self.grid.all_islands),
            )

    def update_database(
        self,
        new_program_code: str,
        parent_idx: int,
        islands: list[int],
        evaluate_return: EvalReturn,
    ) -> None:
        code_diversity = self.evaluator.code_diversity(
            new_program_code, comparison_files=self.grid.get_all_code_files()
        )
        program_length = self.evaluator.program_length(new_program_code)
        if evaluate_return.valid:
            fitness = evaluate_return.fitness
        else:
            fitness = INVALID_FITNESS

        program = Program(
            code=new_program_code,
            fitness=fitness,
            metrics=Metrics(
                program_length=program_length,
                code_diversity=code_diversity,
            ),
            eval_return=evaluate_return.model_dump(),
            parent_idx=parent_idx,
        )

        self.grid.attempt_to_replace_program(
            new_program=program,
            islands=islands,
            metrics=(code_diversity, program_length),
        )

        self.grid.maybe_migrate_program(
            program,
        )


# The following functions are taken from Openevolve
def apply_diff(original_code: str, diff_text: str) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        Modified code
    """

    def extract_diffs(diff_text: str) -> list[tuple[str, str]]:
        """
        Extract diff blocks from the diff text

        Args:
            diff_text: Diff in the SEARCH/REPLACE format

        Returns:
            List of tuples (search_text, replace_text)
        """
        diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
        return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]

    # Split into lines for easier processing
    original_lines = original_code.split("\n")
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text)

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        # Find where the search pattern starts in the original code
        for i in range(len(result_lines) - len(search_lines) + 1):
            if result_lines[i : i + len(search_lines)] == search_lines:
                # Replace the matched section
                result_lines[i : i + len(search_lines)] = replace_lines
                break

    return "\n".join(result_lines)
