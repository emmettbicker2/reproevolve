import asyncio
import importlib.util
import os
import sys
import tempfile
from typing import Sequence

from pydantic import BaseModel


class EvalReturn(BaseModel):
    fitness: float
    validity: float
    eval_time: float

    @property
    def valid(self) -> bool:
        return self.validity == 1.0


class Evaluator:
    def __init__(self, evaluator_program_path: str, timeout_s: int):
        self.timeout_s = timeout_s
        self.load_evaluation_function(evaluator_program_path)

    def load_evaluation_function(self, evaluator_program_path: str) -> None:
        """Load the evaluation function from the evaluation file"""
        if not os.path.exists(evaluator_program_path):
            raise ValueError(f"Evaluation file {evaluator_program_path} not found")

        eval_dir = os.path.dirname(os.path.abspath(evaluator_program_path))
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)

        spec = importlib.util.spec_from_file_location(
            "evaluation_module", evaluator_program_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec from {evaluator_program_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["evaluation_module"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "evaluate"):
            raise AttributeError(
                f"Evaluation file {evaluator_program_path} does not contain an 'evaluate' function"
            )

        # Loading successful
        self.evaluate_function = module.evaluate

    async def run_evaluation(self, code: str) -> EvalReturn:
        """
        Directly evaluate a program using the evaluation function with timeout

        Args:
            program_path: Path to the program file

        Returns:
            Dictionary of metrics or EvaluationResult with metrics and artifacts

        Raises:
            asyncio.TimeoutError: If evaluation exceeds timeout
            Exception: If evaluation function raises an exception
        """

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as program_path:
            program_path.write(code.encode())

            # Create a coroutine that runs the evaluation function in an executor
            async def run_evaluation():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self.evaluate_function, program_path.name
                )

        # Run the evaluation with timeout - let exceptions bubble up for retry handling
        try:
            result = await asyncio.wait_for(run_evaluation(), timeout=self.timeout_s)
        except TimeoutError as e:
            print(f"TimeoutError: {e}")
            return EvalReturn(fitness=0.0, validity=0.0, eval_time=self.timeout_s)

        # Return result as-is to be processed by _process_evaluation_result
        # This supports both dict and EvaluationResult returns, just like _cascade_evaluate
        return EvalReturn.model_validate(result)

    @staticmethod
    def program_length(code: str) -> float:
        """All metrics are floats in my implementation, so I cast to float"""
        return float(len(code))

    @staticmethod
    def code_diversity(code: str, comparison_files: Sequence[str]) -> float:
        if len(comparison_files) == 0:
            return 0.0

        diversity_scores = [
            Evaluator._diversity_between_two_files(code, file)
            for file in comparison_files
        ]
        print("Diversity scores:", sum(diversity_scores) / len(diversity_scores))
        return sum(diversity_scores) / len(diversity_scores)

    @staticmethod
    def _diversity_between_two_files(
        file_contents_1: str, file_contents_2: str
    ) -> float:
        repo_string_1 = "\n".join(file_contents_1)
        repo_string_2 = "\n".join(file_contents_2)

        if repo_string_1 == repo_string_2:
            return 0.0

        # Length difference (scaled to reasonable range)
        len1, len2 = len(repo_string_1), len(repo_string_2)
        length_diff = abs(len1 - len2)

        # Line count difference
        lines1 = repo_string_1.count("\n")
        lines2 = repo_string_2.count("\n")
        line_diff = abs(lines1 - lines2)

        # Simple character set difference
        chars1 = set(repo_string_1)
        chars2 = set(repo_string_2)
        char_diff = len(chars1.symmetric_difference(chars2))

        # Combine metrics (scaled to match original edit distance range)
        diversity = length_diff * 0.1 + line_diff * 10 + char_diff * 0.5
        return diversity
