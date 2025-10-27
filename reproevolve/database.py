from pydantic import BaseModel
from typing import Any, Optional
import random
from typing import Sequence


class Metrics(BaseModel):
    code_diversity: float
    program_length: float

    def __str__(self):
        return f"Code Diversity: {round(self.code_diversity, 3)},  Program Length: {self.program_length}"


class Program(BaseModel):
    code: str
    fitness: float
    metrics: Metrics
    eval_return: dict[str, Any]
    """The exact return from the evaluation function"""
    parent_idx: int
    """This is useful to keep as an attribute of program, but not sure if it should go in _GridEntry"""


class _GridEntry(BaseModel):
    program: Program
    index: int
    islands: list[int]


# Our grid can theoretically be N-D, but this codebase just implements a 2D grid
GRID_DIMENSIONS = 2
ProgramGrid2D = list[list[Optional[_GridEntry]]]


class Grid:
    """This is the MAP-Elites grid"""

    grid: ProgramGrid2D
    grid_size: int
    """Our grid has dimensions grid_size x grid_size"""
    grid_entry_index: int

    migration_rate: float
    """Rate at which to migrate from one island to another"""

    def __init__(self, grid_size: int, num_islands: int, migration_rate: float) -> None:
        """Grid size is the number"""
        self.grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        self.all_islands = list(range(num_islands))
        self.grid_size = grid_size
        self.grid_entry_index = 0
        self.migration_rate = migration_rate

    def _get_all_grid_entries(self, islands: int | list[int]) -> Sequence[_GridEntry]:
        if isinstance(islands, int):
            islands = [islands]

        programs: list[_GridEntry] = []

        for row in self.grid:
            for program in row:
                if program is not None and any(
                    island in program.islands for island in islands
                ):
                    programs.append(program)

        return programs

    def _get_all_grid_entries_with_exclusions(
        self, islands: int | list[int], exclude_programs: Sequence[Program]
    ) -> Sequence[_GridEntry]:
        return [
            g
            for g in self._get_all_grid_entries(islands)
            if g.program not in exclude_programs
        ]

    def _get_all_programs_with_exclusions(
        self, islands: int | list[int], exclude_programs: Sequence[Program]
    ) -> Sequence[Program]:
        return [
            g.program
            for g in self._get_all_grid_entries(islands)
            if g.program not in exclude_programs
        ]

    def get_all_code_files(self) -> Sequence[str]:
        all_grid_entries = self._get_all_grid_entries(self.all_islands)
        return [g.program.code for g in all_grid_entries]

    def random_program(self, island: int) -> Program:
        grid_entries = self._get_all_grid_entries(island)

        assert len(grid_entries) > 0, (
            "Grid has no programs in it. A starting program should be added."
        )

        return random.choice(grid_entries).program

    def get_index_from_program(self, program: Program) -> int:
        grid_entries = self._get_all_grid_entries(islands=self.all_islands)
        for g in grid_entries:
            if g.program is program:
                return g.index
        raise ValueError("Program not found in the grid")

    def get_program_from_index(self, idx: int) -> Optional[Program]:
        """Returns None if not found"""
        grid_entries = self._get_all_grid_entries(islands=self.all_islands)
        for g in grid_entries:
            if g.index is idx:
                return g.program
        return None

    def random_programs(
        self, n: int, islands: list[int], exclude_programs: Sequence[Program] = []
    ) -> Sequence[Program]:
        """Returns n random programs from the grid"""
        programs = self._get_all_programs_with_exclusions(islands, exclude_programs)
        return random.sample(programs, k=min(n, len(programs)))

    def best_program(self, islands: list[int]) -> Program:
        """Setting island to None gets the best program from every single island"""
        return self.best_programs(n=1, islands=islands)[0]

    def best_programs(
        self, n: int, islands: list[int], exclude_programs: Sequence[Program] = []
    ) -> Sequence[Program]:
        """Returns the top n programs. If there are fewer than n programs, returns all programs
        Setting island to None gets the best programs from every single island"""
        programs = self._get_all_programs_with_exclusions(islands, exclude_programs)
        sorted_programs = sorted(programs, key=lambda x: x.fitness, reverse=True)
        return sorted_programs[:n]

    def recent_programs(
        self, n: int, islands: list[int], exclude_programs: Sequence[Program] = []
    ) -> Sequence[Program]:
        """Returns the most recent n programs. If there are fewer than n programs, returns all programs"""
        grid_entries = self._get_all_grid_entries_with_exclusions(
            islands, exclude_programs
        )
        sorted_grid_entries = sorted(grid_entries, key=lambda x: x.index, reverse=True)
        sorted_programs = [g.program for g in sorted_grid_entries]
        return sorted_programs[:n]

    def get_next_island(self):
        """Returns a random island"""

        return random.choice(self.all_islands)

    def get_coordinates(self, metrics: tuple[float, float]) -> tuple[int, int]:
        grid_entries = self._get_all_grid_entries(islands=self.all_islands)
        programs = [g.program for g in grid_entries]

        metric_1 = metrics[0]
        metric_2 = metrics[1]

        existing_metric_1_list = [
            program.metrics.code_diversity for program in programs
        ]
        existing_metric_2_list = [
            program.metrics.program_length for program in programs
        ]
        # Use min-max normalization for more stable bucketing
        min_metric_1 = min(existing_metric_1_list + [metric_1])
        max_metric_1 = max(existing_metric_1_list + [metric_1])
        min_metric_2 = min(existing_metric_2_list + [metric_2])
        max_metric_2 = max(existing_metric_2_list + [metric_2])

        # Avoid division by zero
        if max_metric_1 - min_metric_1 > 1e-10:
            normalized_1 = (metric_1 - min_metric_1) / (max_metric_1 - min_metric_1)
        else:
            normalized_1 = 0.5

        if max_metric_2 - min_metric_2 > 1e-10:
            normalized_2 = (metric_2 - min_metric_2) / (max_metric_2 - min_metric_2)
        else:
            normalized_2 = 0.5

        # Map to grid coordinates
        coord_1 = min(int(normalized_1 * self.grid_size), self.grid_size - 1)
        coord_2 = min(int(normalized_2 * self.grid_size), self.grid_size - 1)

        return (coord_1, coord_2)

    def attempt_to_replace_program(
        self, new_program: Program, islands: list[int], metrics: tuple[float, float]
    ) -> Optional[Program]:
        """
        Replaces the program at the specified coordinates
        if the new program has a higher fitness than the current program
        """

        row, column = self.get_coordinates(metrics)
        old_entry = self.grid[row][column]
        if old_entry is None or old_entry.program.fitness <= new_program.fitness:
            print(
                f"New program in ({row}, {column}) with fitness {new_program.fitness}. Metrics: {metrics}"
            )
            self.replace_program(
                islands=islands, row=row, column=column, program=new_program
            )

    def replace_program(
        self, islands: list[int], row: int, column: int, program: Program
    ):
        self.grid[row][column] = _GridEntry(
            program=program,
            index=self.grid_entry_index,
            islands=islands,
        )
        self.grid_entry_index += 1

    def maybe_migrate_program(self, program: Program):
        if not random.random() < self.migration_rate:
            return

        # Locate the program in the grid
        grid_entry = None
        for row in self.grid:
            for item in row:
                if item is not None and item.program is program:
                    grid_entry = item
        if grid_entry is None:
            print("Attempted to migrate but the program never made it in the grid")
            return

        # Filter out the first progrma
        if grid_entry.index == 0:
            print("Attempted to migrate the first program, not going to")
            return

        # Make sure the grid entry only has one island
        assert len(grid_entry.islands) == 1

        # Determine the new island to move to
        offset = random.choice([-1, 1])
        island = grid_entry.islands[0]
        new_island = (island + offset) % len(self.all_islands)

        # Log
        print(f"Migrating program from {island} to {new_island}")

        # Migrate
        grid_entry.islands.append(new_island)
