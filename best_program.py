# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using numerical optimization."""

import numpy as np
from scipy.optimize import minimize

# Tuple of (centers, radii, sum_of_radii)
# centers: np.array of shape (26, 2) with (x, y) coordinates
# radii: np.array of shape (26) with radius of each circle
# sum_of_radii: Sum of all radii
ReturnType = tuple[np.ndarray, np.ndarray, float]


def objective_function(flat_centers: np.ndarray, n: int) -> float:
    """
    Enhanced objective function with weighted optimization to favor larger central circles.
    flat_centers: 1D array of [x0, y0, x1, y1, ...]
    """
    centers = flat_centers.reshape((n, 2))

    # Calculate radii based on the current fixed center configuration
    radii = compute_max_radii(centers)

    # Calculate distance from center for each circle
    center_point = np.array([0.5, 0.5])
    distances_from_center = np.sqrt(np.sum((centers - center_point) ** 2, axis=1))

    # Apply weighting: favor larger circles near center, smaller at edges
    # This follows mathematical principles of optimal circle packing
    weights = 1.0 - 0.3 * (distances_from_center / 0.5)
    weighted_sum = np.sum(radii * weights)

    # Combined objective: 85% regular sum, 15% weighted sum
    combined_objective = 0.85 * np.sum(radii) + 0.15 * weighted_sum

    return -combined_objective  # Minimize negative combined objective


def compute_max_radii(centers: np.ndarray) -> np.ndarray:
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square using iterative relaxation.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]

    # 1. Initial limit by square borders
    radii = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))

    # 2. Iterative adjustment based on circle overlap (Relaxation)
    max_iter = 150  # Increased iterations slightly

    # Pre-calculate all pairwise distances once for efficiency
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)

    for _ in range(max_iter):
        changed = False

        for i in range(n):
            # Constraint 1: Boundary limit (re-check)
            r_boundary = min(
                centers[i, 0], centers[i, 1], 1 - centers[i, 0], 1 - centers[i, 1]
            )

            # Constraint 2: Overlap limits
            r_overlap_max = float("inf")
            for j in range(n):
                if i == j:
                    continue

                dist = distances[i, j]
                # Max radius for i such that r_i + r_j <= dist
                r_overlap_max = min(r_overlap_max, dist - radii[j])

            new_radius = min(r_boundary, r_overlap_max)

            # Use a small tolerance to detect convergence
            if abs(radii[i] - new_radius) > 1e-13:
                radii[i] = max(0.0, new_radius)
                changed = True

        if not changed:
            break

    return radii


def construct_packing() -> ReturnType:
    """
    Optimize the placement of 26 circles within a unit square
    to maximize the sum of their radii using multi-start optimization.
    """
    n = 26

    # --- 1. Define multiple initial guesses based on successful historical structures ---

    def create_config_boundary_dense():
        # Configuration emphasizing boundary placement with precise positioning
        centers = np.zeros((n, 2))
        d = 0.14  # Corner distance from edge
        centers[0:4] = [[d, d], [1 - d, d], [d, 1 - d], [1 - d, 1 - d]]
        e = 0.07  # Edge distance from boundary
        centers[4:8] = [[0.5, e], [0.5, 1 - e], [e, 0.5], [1 - e, 0.5]]
        centers[8] = [0.5, 0.5]  # Center
        R1 = 0.28  # Inner ring radius
        for i in range(8):
            angle = 2 * np.pi * i / 8
            centers[i + 9] = [0.5 + R1 * np.cos(angle), 0.5 + R1 * np.sin(angle)]
        R2 = 0.62  # Outer ring radius
        for i in range(9):
            angle = 2 * np.pi * i / 9 + np.pi / 9  # Staggered offset
            centers[i + 17] = [0.5 + R2 * np.cos(angle), 0.5 + R2 * np.sin(angle)]
        return centers

    def create_config_variable_ring():
        # Configuration with variable radii and non-uniform spacing
        centers = np.zeros((n, 2))
        d = 0.12
        centers[0:4] = [[d, d], [1 - d, d], [d, 1 - d], [1 - d, 1 - d]]
        e = 0.12
        centers[4:8] = [[0.5, e], [0.5, 1 - e], [e, 0.5], [1 - e, 0.5]]
        centers[8] = [0.5, 0.5]
        R1 = 0.28
        for i in range(8):
            angle = 2 * np.pi * i / 8
            centers[i + 9] = [0.5 + R1 * np.cos(angle), 0.5 + R1 * np.sin(angle)]
        R2 = 0.62
        for i in range(9):
            angle = 2 * np.pi * i / 9 + np.pi / 18  # Different offset
            centers[i + 17] = [0.5 + R2 * np.cos(angle), 0.5 + R2 * np.sin(angle)]
        return centers

    def create_config_asymmetric_ring():
        # Non-uniform ring structure with asymmetric positioning
        centers = np.zeros((n, 2))
        d = 0.13
        centers[0:4] = [[d, d], [1 - d, d], [d, 1 - d], [1 - d, 1 - d]]
        e = 0.06
        centers[4:8] = [[0.5, e], [0.5, 1 - e], [e, 0.5], [1 - e, 0.5]]
        centers[8] = [0.5, 0.5]
        R1 = 0.30
        for i in range(8):
            angle = 2 * np.pi * i / 8 + np.pi / 16
            centers[i + 9] = [0.5 + R1 * np.cos(angle), 0.5 + R1 * np.sin(angle)]
        R2 = 0.68
        for i in range(9):
            angle = 2 * np.pi * i / 9 + np.pi / 18
            centers[i + 17] = [0.5 + R2 * np.cos(angle), 0.5 + R2 * np.sin(angle)]
        return centers

    def create_config_specialized():
        # Specialized configuration with variable-sized circles and strategic positioning
        centers = np.zeros((n, 2))

        # Corner circles with precise positioning
        corner_dist = 0.105
        centers[0:4] = [
            [corner_dist, corner_dist],
            [1 - corner_dist, corner_dist],
            [corner_dist, 1 - corner_dist],
            [1 - corner_dist, 1 - corner_dist],
        ]

        # Edge circles with optimal distance
        edge_dist = 0.065
        centers[4:8] = [
            [0.5, edge_dist],
            [1 - edge_dist, 0.5],
            [0.5, 1 - edge_dist],
            [edge_dist, 0.5],
        ]

        # Center circle
        centers[8] = [0.5, 0.5]

        # First ring - 8 circles in optimal arrangement
        r1 = 0.265
        for i in range(8):
            angle = 2 * np.pi * i / 8 + np.pi / 16  # Slight offset
            centers[i + 9] = [0.5 + r1 * np.cos(angle), 0.5 + r1 * np.sin(angle)]

        # Outer ring - 9 circles with variable spacing
        for i in range(9):
            angle = 2 * np.pi * i / 9 + np.pi / 12  # Different offset
            r2 = 0.66 + 0.03 * np.sin(3 * angle)  # Variable radius to break symmetry
            centers[i + 17] = [0.5 + r2 * np.cos(angle), 0.5 + r2 * np.sin(angle)]

        return centers

    initial_structures = [
        create_config_boundary_dense(),
        create_config_variable_ring(),
        create_config_asymmetric_ring(),
        create_config_specialized(),
    ]

    # --- 2. Define optimization bounds ---
    epsilon = 0.005
    bounds = [(epsilon, 1.0 - epsilon) for _ in range(n * 2)]

    # --- 3. Run multi-start optimization ---
    best_result = (None, None, -1.0)  # (centers, radii, sum_radii)

    for initial_centers in initial_structures:
        initial_params = initial_centers.flatten()

        # Phase 1: L-BFGS-B
        result1 = minimize(
            objective_function,
            initial_params,
            args=(n,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1500, "ftol": 1e-9},
        )

        # Phase 2: SLSQP refinement
        result2 = minimize(
            objective_function,
            result1.x,
            args=(n,),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-11},
        )

        # Calculate radii and sum
        optimized_centers = result2.x.reshape((n, 2))
        radii = compute_max_radii(optimized_centers)

        # Post-optimization refinement: Enhanced Greedy Growth with Perturbation
        improved = True
        iteration = 0
        max_iterations = 60  # High iteration count for thorough growth

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Grow smallest circles first
            indices = np.argsort(radii)

            for idx in indices:
                original_radius = radii[idx]
                test_radius = original_radius * 1.0022  # More aggressive growth factor

                # Boundary check
                x, y = optimized_centers[idx]
                if test_radius > min(x, y, 1 - x, 1 - y) + 1e-9:
                    continue

                # Overlap check
                valid = True
                for j in range(n):
                    if j == idx:
                        continue
                    dist = np.sqrt(
                        np.sum((optimized_centers[idx] - optimized_centers[j]) ** 2)
                    )
                    if test_radius + radii[j] > dist + 1e-9:
                        valid = False
                        break

                if valid:
                    radii[idx] = test_radius
                    improved = True

            # Add perturbation phase every 10 iterations to escape local optima
            if iteration % 10 == 0 and iteration >= 20:
                # Save current best state
                best_centers = optimized_centers.copy()
                best_radii = radii.copy()
                best_sum = np.sum(radii)

                # Try perturbing each circle slightly
                for idx in range(n):
                    # Skip largest circles to maintain stability
                    if radii[idx] > np.median(radii):
                        continue

                    # Try small perturbations in 4 directions
                    for direction in [(0.003, 0), (-0.003, 0), (0, 0.003), (0, -0.003)]:
                        # Save original position
                        original_pos = optimized_centers[idx].copy()

                        # Apply perturbation
                        optimized_centers[idx] += direction
                        optimized_centers[idx] = np.clip(
                            optimized_centers[idx], epsilon, 1.0 - epsilon
                        )

                        # Recalculate radii
                        new_radii = compute_max_radii(optimized_centers)

                        # If better, keep it and continue growth
                        if np.sum(new_radii) > best_sum:
                            best_centers = optimized_centers.copy()
                            best_radii = new_radii.copy()
                            best_sum = np.sum(new_radii)
                            improved = True
                        else:
                            # Restore original position
                            optimized_centers[idx] = original_pos

                # Restore best state found during perturbation
                optimized_centers = best_centers
                radii = best_radii

        # Final refinement: Optimize each circle position individually
        for _ in range(2):  # Just a few iterations to avoid timeout
            for idx in range(n):
                original_pos = optimized_centers[idx].copy()
                original_sum = np.sum(radii)

                # Try small movements in 8 directions
                for dx in [-0.005, 0, 0.005]:
                    for dy in [-0.005, 0, 0.005]:
                        if dx == 0 and dy == 0:
                            continue

                        # Try this position
                        test_pos = original_pos + np.array([dx, dy])
                        test_pos = np.clip(test_pos, epsilon, 1.0 - epsilon)

                        optimized_centers[idx] = test_pos
                        test_radii = compute_max_radii(optimized_centers)

                        # Keep if better
                        if np.sum(test_radii) > original_sum:
                            radii = test_radii
                            original_sum = np.sum(radii)
                        else:
                            # Revert position
                            optimized_centers[idx] = original_pos

        sum_radii = np.sum(radii)

        # Keep the best result
        if sum_radii > best_result[2]:
            best_result = (optimized_centers, radii, sum_radii)

    return best_result


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing() -> ReturnType:
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers: np.ndarray, radii: np.ndarray) -> None:
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
