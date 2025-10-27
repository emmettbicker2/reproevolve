import random
from reproevolve.database import Grid, Program
from typing import Any, Sequence
from reproevolve.prompt_templates import (
    EVOLUTION_HISTORY_TEMPLATE,
    PREVIOUS_ATTEMPT_TEMPLATE,
    TOP_PROGRAM_TEMPLATE,
    DIFF_USER_TEMPLATE,
)


N_TOP_PROGRAMS = 3
N_DIVERSE_PROGRAMS = 3
INVALID_FITNESS = 0.0
SUGGEST_SIMPLIFICATION_AFTER_CHARS = 500


class PromptBuilder:
    grid: Grid

    def __init__(self, grid: Grid):
        self.grid = grid

    def get_user_prompt(self, program: Program, island: int) -> str:
        previous_programs = self.grid.best_programs(n=N_TOP_PROGRAMS, islands=[island])
        top_programs = self.grid.best_programs(
            n=N_TOP_PROGRAMS + N_DIVERSE_PROGRAMS, islands=[island]
        )

        evolution_history = self._format_evolution_history(
            previous_programs, top_programs
        )
        metrics = self._format_metrics(program.eval_return)
        improvement_areas = self._identify_improvement_areas(program, previous_programs)

        return DIFF_USER_TEMPLATE.format(
            metrics=metrics,
            improvement_areas=improvement_areas,
            evolution_history=evolution_history,
            current_program=program.code,
        )

    def get_evolution_history(self, island: int) -> str:
        """Format the evolution history for the prompt"""
        previous_programs = self.grid.best_programs(n=N_TOP_PROGRAMS, islands=[island])
        top_programs = self.grid.best_programs(
            n=N_TOP_PROGRAMS + N_DIVERSE_PROGRAMS, islands=[island]
        )

        return self._format_evolution_history(
            previous_programs=previous_programs,
            top_programs=top_programs,
        )

    def _format_metrics(self, metrics: dict[str, float]) -> str:
        """Format metrics for the prompt using safe formatting"""
        # Use safe formatting to handle mixed numeric and string values
        formatted_parts: list[str] = []
        for name, value in metrics.items():
            formatted_parts.append(f"- {name}: {value:.4f}")
        return "\n".join(formatted_parts)

    def _identify_improvement_areas(
        self,
        current_program: Program,
        previous_programs: Sequence[Program],
    ) -> str:
        """Identify potential areas for improvement"""

        improvement_areas: list[str] = []

        # Check program length
        if len(current_program.code) > SUGGEST_SIMPLIFICATION_AFTER_CHARS:
            improvement_areas.append(
                "Consider simplifying the code to improve readability and maintainability"
            )

        # Check for performance patterns in previous attempts
        if len(previous_programs) >= 2:
            recent_attempts = previous_programs[-2:]
            metrics_improved: list[str] = []
            metrics_regressed: list[str] = []
            metrics = current_program.eval_return

            for metric, value in metrics.items():
                improved = True
                regressed = True

                for attempt in recent_attempts:
                    attempt_value = attempt.eval_return.get(metric, 0)
                    # Only compare if both values are numeric
                    if attempt_value <= value:
                        regressed = False
                    if attempt_value >= value:
                        improved = False

                if improved and metric not in metrics_improved:
                    metrics_improved.append(metric)
                if regressed and metric not in metrics_regressed:
                    metrics_regressed.append(metric)

            if metrics_improved:
                improvement_areas.append(
                    f"Metrics showing improvement: {', '.join(metrics_improved)}. "
                    "Consider continuing with similar changes."
                )

            if metrics_regressed:
                improvement_areas.append(
                    f"Metrics showing regression: {', '.join(metrics_regressed)}. "
                    "Consider reverting or revising recent changes in these areas."
                )

        # If we don't have specific improvements to suggest
        if not improvement_areas:
            improvement_areas.append(
                "Focus on optimizing the code for better performance on the target metrics"
            )

        return "\n".join([f"- {area}" for area in improvement_areas])

    def _format_evolution_history(
        self,
        previous_programs: Sequence[Program],
        top_programs: Sequence[Program],
        language: str = "python",
    ) -> str:
        """Format the evolution history for the prompt"""
        # Get templates
        history_template = EVOLUTION_HISTORY_TEMPLATE
        previous_attempt_template = PREVIOUS_ATTEMPT_TEMPLATE
        top_program_template = TOP_PROGRAM_TEMPLATE

        # Format previous attempts (most recent first)
        previous_attempts_str = ""
        selected_previous = previous_programs[-min(3, len(previous_programs)) :]

        for i, program in enumerate(reversed(selected_previous)):
            attempt_number = len(previous_programs) - i
            changes = "Unknown changes"

            # Format performance metrics using safe formatting
            performance_parts: list[str] = []
            program_metrics_with_fitness = program.eval_return
            for name, value in program_metrics_with_fitness.items():
                if isinstance(value, (int, float)):
                    try:
                        performance_parts.append(f"{name}: {value:.4f}")
                    except (ValueError, TypeError):
                        performance_parts.append(f"{name}: {value}")
                else:
                    performance_parts.append(f"{name}: {value}")
            performance_str = ", ".join(performance_parts)

            parent_program = self.grid.get_program_from_index(program.parent_idx)
            parent_metrics_with_fitness = (
                parent_program.eval_return if parent_program is not None else {}
            )
            # Determine outcome based on comparison with parent (only numeric metrics)
            outcome = "Mixed results"

            # Check if all numeric metrics improved
            numeric_comparisons_improved: list[bool] = []
            numeric_comparisons_regressed: list[bool] = []

            for m in program_metrics_with_fitness:
                prog_value = program_metrics_with_fitness.get(m, 0)
                parent_value = parent_metrics_with_fitness.get(m, 0)

                # Only compare if both values are numeric
                if isinstance(prog_value, (int, float)) and isinstance(
                    parent_value, (int, float)
                ):
                    if prog_value > parent_value:
                        numeric_comparisons_improved.append(True)
                    else:
                        numeric_comparisons_improved.append(False)

                    if prog_value < parent_value:
                        numeric_comparisons_regressed.append(True)
                    else:
                        numeric_comparisons_regressed.append(False)

            # Determine outcome based on numeric comparisons
            if numeric_comparisons_improved and all(numeric_comparisons_improved):
                outcome = "Improvement in all metrics"
            elif numeric_comparisons_regressed and all(numeric_comparisons_regressed):
                outcome = "Regression in all metrics"

            previous_attempts_str += (
                previous_attempt_template.format(
                    attempt_number=attempt_number,
                    changes=changes,
                    performance=performance_str,
                    outcome=outcome,
                )
                + "\n\n"
            )

        # Format top programs
        top_programs_str = ""
        selected_top = top_programs[: min(N_TOP_PROGRAMS, len(top_programs))]

        for i, program in enumerate(selected_top):
            # Use the full program code
            program_code = program.code
            program_metrics_with_fitness = program.eval_return

            # Calculate a composite score using safe numeric average
            score = safe_numeric_average(program_metrics_with_fitness)

            # Extract key features (this could be more sophisticated)
            key_features: list[str] = []
            for name, value in program_metrics_with_fitness.items():
                if isinstance(value, (int, float)):
                    try:
                        key_features.append(f"Performs well on {name} ({value:.4f})")
                    except (ValueError, TypeError):
                        key_features.append(f"Performs well on {name} ({value})")
                else:
                    key_features.append(f"Performs well on {name} ({value})")

            key_features_str = ", ".join(key_features)

            top_programs_str += (
                top_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    language=language,
                    program_snippet=program_code,
                    key_features=key_features_str,
                )
                + "\n\n"
            )

        # Format diverse programs using num_diverse_programs config
        diverse_programs_str = ""
        if N_DIVERSE_PROGRAMS > 0 and len(top_programs) > N_DIVERSE_PROGRAMS:
            # Skip the top programs we already included
            remaining_programs = top_programs[N_TOP_PROGRAMS:]

            # Sample diverse programs from the remaining
            num_diverse = min(N_DIVERSE_PROGRAMS, len(remaining_programs))
            if num_diverse > 0:
                # Use random sampling to get diverse programs
                diverse_programs = random.sample(remaining_programs, num_diverse)

                diverse_programs_str += "\n\n## Diverse Programs\n\n"

                for i, program in enumerate(diverse_programs):
                    # Use the full program code
                    program_code = program.code
                    program_metrics_with_fitness = program.eval_return

                    # Calculate a composite score using safe numeric average
                    score = safe_numeric_average(program_metrics_with_fitness)

                    # Extract key features

                    key_features = [
                        f"Alternative approach to {name}"
                        for name in list(program_metrics_with_fitness.keys())[
                            :2
                        ]  # Just first 2 metrics
                    ]

                    key_features_str = ", ".join(key_features)

                    diverse_programs_str += (
                        top_program_template.format(
                            program_number=f"D{i + 1}",
                            score=f"{score:.4f}",
                            language=language,
                            program_snippet=program_code,
                            key_features=key_features_str,
                        )
                        + "\n\n"
                    )

        # Combine top and diverse programs
        combined_programs_str = top_programs_str + diverse_programs_str

        # Combine into full history
        return history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=combined_programs_str.strip(),
        )


def safe_numeric_average(metrics: dict[str, Any]) -> float:
    """
    Calculate the average of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Average of numeric values, or 0.0 if no numeric values found
    """
    if not metrics:
        return 0.0

    numeric_values: list[float | int] = []
    for value in metrics.values():
        if isinstance(value, (int, float)):
            try:
                # Convert to float and check if it's a valid number
                float_val = float(value)
                if not (float_val != float_val):  # Check for NaN (NaN != NaN is True)
                    numeric_values.append(float_val)
            except (ValueError, TypeError, OverflowError):
                # Skip invalid numeric values
                continue

    if not numeric_values:
        return 0.0

    return sum(numeric_values) / len(numeric_values)
