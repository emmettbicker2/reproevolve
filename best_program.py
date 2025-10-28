# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

import numpy as np

# Tuple of (centers, radii, sum_of_radii)
# centers: np.array of shape (26, 2) with (x, y) coordinates
# radii: np.array of shape (26) with radius of each circle
# sum_of_radii: Sum of all radii
ReturnType = tuple[np.ndarray, np.ndarray, float]


def construct_packing() -> ReturnType:
    """
    Optimize the placement of 26 circles within a unit square
    to maximize the sum of their radii using numerical optimization.
    """
    from scipy.optimize import minimize
    
    n = 26
    
    # --- 1. Define Initial Guesses (Multi-Start Strategy) ---
    def create_config_boundary_heavy():
        # Structure with boundary circles + inner ring (Historical Best structure)
        centers = np.zeros((n, 2))
        corner_r_dist = 0.09
        centers[0:4] = [[corner_r_dist, corner_r_dist], [corner_r_dist, 1.0 - corner_r_dist], 
                        [1.0 - corner_r_dist, corner_r_dist], [1.0 - corner_r_dist, 1.0 - corner_r_dist]]
        edge_r_dist = 0.05
        centers[4:8] = [[0.5, edge_r_dist], [1.0 - edge_r_dist, 0.5], 
                        [0.5, 1.0 - edge_r_dist], [edge_r_dist, 0.5]]
        centers[8] = [0.5, 0.5] # Center
        
        remaining_start_idx = 9
        remaining_n = n - remaining_start_idx # 17
        intermediate_ring_r = 0.35
        for i in range(remaining_n):
            angle = 2 * np.pi * i / remaining_n
            centers[i + remaining_start_idx] = [0.5 + intermediate_ring_r * np.cos(angle), 
                                                0.5 + intermediate_ring_r * np.sin(angle)]
        return centers

    def create_config_hex_cluster():
        # Hexagonal central cluster (1+6+19 structure attempt)
        centers = np.zeros((n, 2))
        centers[0] = [0.5, 0.5] # Center
        
        # First ring: 6 circles around center in hexagonal arrangement
        r1 = 0.20
        for i in range(6):
            angle = 2 * np.pi * i / 6
            centers[i+1] = [0.5 + r1 * np.cos(angle), 0.5 + r1 * np.sin(angle)]
            
        # Second ring: 19 circles
        r2 = 0.42
        for i in range(19):
            angle = 2 * np.pi * i / 19 + np.pi/19 # Staggered
            centers[i+7] = [0.5 + r2 * np.cos(angle), 0.5 + r2 * np.sin(angle)]
        return centers

    def create_config_grid():
        # Quasi-grid/layered structure
        centers = np.zeros((n, 2))
        grid_size = 5
        spacing = 1.0 / (grid_size + 0.2) 
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n: break
                
                offset = spacing/2 if i % 2 == 1 else 0
                
                x = spacing/2 + j * spacing + offset
                y = spacing/2 + i * spacing
                
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                centers[count] = [x, y]
                count += 1
        
        np.random.seed(10)
        for i in range(count, n):
            centers[i] = np.random.uniform(0.05, 0.95, 2)
            
        return centers

    def create_config_hybrid():
        # Hybrid approach with variable sizes - larger circles at corners and center
        centers = np.zeros((n, 2))
        
        # Place 4 circles at corners with specific offsets
        corner_dist = 0.12
        centers[0:4] = [[corner_dist, corner_dist], 
                        [corner_dist, 1.0 - corner_dist], 
                        [1.0 - corner_dist, corner_dist], 
                        [1.0 - corner_dist, 1.0 - corner_dist]]
        
        # Place 4 circles at edge midpoints
        centers[4:8] = [[0.5, 0.08], 
                        [0.92, 0.5], 
                        [0.5, 0.92], 
                        [0.08, 0.5]]
        
        # Center circle
        centers[8] = [0.5, 0.5]
        
        # Inner ring of 8 circles
        inner_r = 0.27
        for i in range(8):
            angle = 2 * np.pi * i / 8
            centers[9 + i] = [0.5 + inner_r * np.cos(angle), 
                             0.5 + inner_r * np.sin(angle)]
        
        # Outer ring of 9 circles
        outer_r = 0.46
        for i in range(9):
            angle = 2 * np.pi * i / 9 + np.pi/9  # Staggered from inner ring
            centers[17 + i] = [0.5 + outer_r * np.cos(angle), 
                              0.5 + outer_r * np.sin(angle)]
        
        return centers
    
    initial_configs = [create_config_boundary_heavy(), create_config_hex_cluster(), 
                       create_config_grid(), create_config_hybrid()]
    
    # --- 2. Define Objective Function ---
    bound_val = 0.005
    bounds = [(bound_val, 1.0 - bound_val) for _ in range(n * 2)]
    
    def objective(params):
        centers = params.reshape((n, 2))
        radii = compute_max_radii(centers)
        
        if np.any(radii < -1e-9): 
             return 1e10
        
        sum_radii = np.sum(radii)
        
        # WEIGHTED OBJECTIVE: Bias center growth. Larger radii near the center are weighted higher.
        center_point = np.array([0.5, 0.5])
        distances_from_center = np.sqrt(np.sum((centers - center_point) ** 2, axis=1))
        # Weight factor: 1.0 at center, reducing to 0.7 at the edge of the initial placement radius (approx 0.46)
        weights = 1.0 - 0.4 * (distances_from_center / 0.5) 
        weighted_sum = np.sum(radii * weights)
        
        # Combine objectives: 80% standard sum, 20% weighted sum
        combined_objective = 0.80 * sum_radii + 0.20 * weighted_sum
        
        return -combined_objective # Minimize negative combined objective
    
    # --- 3. Multi-Start Optimization Loop ---
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    MAX_ITER_PER_START = 3000
    FTOL_TOLERANCE = 1e-11 

    for initial_centers in initial_configs:
        initial_params = initial_centers.flatten()
        
        result = minimize(
            objective, 
            initial_params, 
            method='SLSQP', 
            bounds=bounds,
            options={'maxiter': MAX_ITER_PER_START, 'ftol': FTOL_TOLERANCE}
        )
        
        # Recalculate true sum for comparison, as the objective is weighted
        current_centers = result.x.reshape((n, 2))
        current_radii = compute_max_radii(current_centers)
        current_sum = np.sum(current_radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = current_centers
            best_radii = current_radii
            
    # --- 4. Iterative Refinement with Multiple Optimization Stages ---
    # First stage: Standard optimization
    final_result = minimize(
        objective, 
        best_centers.flatten(), 
        method='SLSQP', 
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-12}
    )
    
    optimized_centers = final_result.x.reshape((n, 2))
    current_radii = compute_max_radii(optimized_centers)
    current_sum = np.sum(current_radii)
    
    # Second stage: Targeted refinement focusing on smallest circles
    for refinement_round in range(3):
        # Identify smallest circles to focus optimization on
        smallest_indices = np.argsort(current_radii)[:10]
        
        # Create a mask for parameters to optimize (only smallest circles)
        mask = np.zeros(n * 2, dtype=bool)
        for idx in smallest_indices:
            mask[idx*2] = True    # x coordinate
            mask[idx*2+1] = True  # y coordinate
        
        # Create reduced parameter vector and bounds
        reduced_params = final_result.x[mask]
        reduced_bounds = [bounds[i] for i, m in enumerate(mask) if m]
        
        # Define reduced objective function
        def reduced_objective(reduced_params):
            full_params = final_result.x.copy()
            full_params[mask] = reduced_params
            centers = full_params.reshape((n, 2))
            radii = compute_max_radii(centers)
            # Re-use the weighted objective for refinement stability
            sum_radii = np.sum(radii)
            center_point = np.array([0.5, 0.5])
            distances_from_center = np.sqrt(np.sum((centers - center_point) ** 2, axis=1))
            weights = 1.0 - 0.4 * (distances_from_center / 0.5) 
            weighted_sum = np.sum(radii * weights)
            combined_objective = 0.80 * sum_radii + 0.20 * weighted_sum
            return -combined_objective
        
        # Run targeted optimization
        reduced_result = minimize(
            reduced_objective,
            reduced_params,
            method='SLSQP',
            bounds=reduced_bounds,
            options={'maxiter': 1000, 'ftol': 1e-13}
        )
        
        # Update full parameter vector with optimized values
        full_params = final_result.x.copy()
        full_params[mask] = reduced_result.x
        
        # Check if improvement was achieved
        new_centers = full_params.reshape((n, 2))
        new_radii = compute_max_radii(new_centers)
        new_sum = np.sum(new_radii)
        
        if new_sum > current_sum:
            optimized_centers = new_centers
            current_radii = new_radii
            current_sum = new_sum
            final_result.x = full_params
    
    final_radii = current_radii
    
    # --- 5. Post-Optimization Greedy Growth Phase with Adaptive Growth Rate ---
    improved = True
    iteration = 0
    max_iterations = 50
    
    # Pre-compute distances for efficiency in the growth phase
    distances = np.sqrt(np.sum((optimized_centers[:, None, :] - optimized_centers[None, :, :]) ** 2, axis=-1))
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # WEIGHTED OBJECTIVE: Bias center growth. Larger radii near the center are weighted higher.
        growth_factor = 1.0 + (0.008 * (max_iterations - iteration) / max_iterations)
        
        # Always prioritize smallest circles for growth later in the process, as they are most constrained
        indices = np.argsort(final_radii)   # Ascending order (smallest first)
        
        for idx in indices:
            original_radius = final_radii[idx]
            test_radius = original_radius * growth_factor
            
            # Boundary check
            x, y = optimized_centers[idx]
            if test_radius > min(x, y, 1 - x, 1 - y) + 1e-9:
                continue
                
            # Overlap check using pre-computed distances
            valid = True
            for j in range(n):
                if j == idx: continue
                if test_radius + final_radii[j] > distances[idx, j] + 1e-9:
                    valid = False
                    break
            
            if valid:
                final_radii[idx] = test_radius
                improved = True
                
    final_sum_radii = np.sum(final_radii)
    
    return optimized_centers, final_radii, final_sum_radii


def compute_max_radii(centers: np.ndarray) -> np.ndarray:
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    Uses accelerated iterative relaxation for true maximization with fixed centers.
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # 1. Initial limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # 2. Pre-calculate all pairwise distances for efficiency
    distances = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=-1))
    
    # 3. Iterative adjustment with prioritization and acceleration
    max_iter = 100
    convergence_threshold = 1e-13  # Tighter convergence
    
    # Sort indices by distance to center of square for better convergence pattern
    center_dists = np.sqrt(np.sum((centers - 0.5) ** 2, axis=1))
    sorted_indices = np.argsort(center_dists)
    
    for iteration in range(max_iter):
        max_change = 0.0
        
        # Process circles from outside in (better convergence pattern)
        for idx in sorted_indices:
            i = idx
            # Boundary constraint
            r_boundary = min(centers[i, 0], centers[i, 1], 1 - centers[i, 0], 1 - centers[i, 1])
            
            # Overlap constraints - find the most limiting one
            r_overlap_max = float('inf')
            
            for j in range(n):
                if i == j:
                    continue
                
                dist = distances[i, j]
                r_limit = dist - radii[j]
                
                if r_limit < r_overlap_max:
                    r_overlap_max = r_limit
            
            # Apply the most restrictive constraint
            new_radius = min(r_boundary, r_overlap_max)
            
            # Track maximum change for convergence check
            change = abs(radii[i] - new_radius)
            max_change = max(max_change, change)
            
            # Update radius
            if change > convergence_threshold:
                radii[i] = max(0.0, new_radius)
        
        # Check for convergence using maximum change
        if max_change <= convergence_threshold:
            break
            
    # 4. Final validation pass to ensure no overlaps
    TOL = 1e-11
    for i in range(n):
        # Check boundary constraints
        r_boundary = min(centers[i, 0], centers[i, 1], 1 - centers[i, 0], 1 - centers[i, 1])
        radii[i] = min(radii[i], r_boundary)
        
        # Check overlap constraints
        for j in range(n):
            if i == j:
                continue
            dist = distances[i, j]
            if radii[i] + radii[j] > dist + TOL:
                radii[i] = dist - radii[j] - TOL
    
    return radii


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
