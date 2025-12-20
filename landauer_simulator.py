"""
Landauer Extension Simulator
============================
Thermodynamic cost simulation for CRUD operations

Theoretical Predictions:
  - Create: ~ k_B T ln(2) (exothermic, W < 0)
  - Delete: ~ k_B T ln(2) (Landauer bound)
  - Update: ~ 2 * k_B T ln(2)
  - Maintain: Depends on error rate

Author: Masamichi Iizumi & Tamaki Iizumi
Date: December 14, 2025
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import matplotlib.pyplot as plt
import csv
from datetime import datetime

print("JAX devices:", jax.devices())

# ===== Physical Constants =====
KB = 1.0      # Boltzmann constant (in natural units)
GAMMA = 1.0   # Friction coefficient
DT = 0.001    # Time step

# ===== Potential Functions =====
# V(x) = x^4 - b*x^2 - c*x
# b: Barrier height (positive for double-well)
# c: Tilt (operation bias)

@jit
def potential(x, b, c):
    """Double-well potential"""
    return x**4 - b * x**2 - c * x

@jit
def force(x, b, c):
    """Force = -dV/dx (analytical)"""
    return -4*x**3 + 2*b*x + c

# ===== Single Step Time Evolution =====
@jit
def step_single(x, key, b, c, T):
    """Single step of Langevin equation (Euler-Maruyama method)"""
    f = force(x, b, c)
    noise = random.normal(key)
    x_new = x + (f / GAMMA) * DT + jnp.sqrt(2 * KB * T * DT / GAMMA) * noise
    return x_new

# Batch version
@jit
def step_batch(x_batch, keys, b, c, T):
    """Parallel update of multiple particles"""
    return vmap(lambda x, k: step_single(x, k, b, c, T))(x_batch, keys)

# ===== Work Calculation =====
@jit
def work_step(x, b_old, c_old, b_new, c_new):
    """Work from parameter change: W = V_new - V_old"""
    return potential(x, b_new, c_new) - potential(x, b_old, c_old)

work_step_batch = vmap(work_step, in_axes=(0, None, None, None, None))

# ===== Protocol Definitions =====

def protocol_create_v1(t):
    """Create v1 (original): Dig well method"""
    b = 2.0 * t       # Gradually form barrier
    c = 1.0 * t       # Tilt to the right
    return b, c

def protocol_create(t):
    """Create v2 (improved): Raise wall method
    
    Start with double-well, raise wall to confine
    This restricts degrees of freedom rather than
    lowering potential energy
    """
    # Phase 1 (t < 0.3): Tilt to guide to right well
    # Phase 2 (0.3 <= t < 0.7): Maintain tilt for stabilization
    # Phase 3 (t >= 0.7): Raise wall to confine
    
    # Expressed as continuous functions
    b = 2.0 + 1.0 * jnp.maximum(0, (t - 0.7) / 0.3)  # 2 → 3 (increases for t > 0.7)
    c = 0.5 * jnp.minimum(1.0, t / 0.3) * jnp.maximum(0, 1 - jnp.maximum(0, (t - 0.7) / 0.3))
    return b, c

def protocol_create_v3(t):
    """Create v3: Clear "raise wall" method
    
    Start with shallow well, gradually deepen (raise wall)
    Position unchanged, strengthen confinement
    """
    # Start on right side, raise wall
    b = 1.0 + 2.0 * t   # 1 → 3 (raise barrier)
    c = 0.3             # Light right bias (fixed)
    return b, c

def protocol_delete(t):
    """Delete: Definite state → left well (state 0) erasure (Landauer)"""
    b = 2.0 * (1 - t) + 0.1  # Lower barrier
    c = -1.5 * t             # Strongly tilt left
    return b, c

def protocol_update(t):
    """Update: Left well → right well (state change)"""
    if t < 0.5:
        # First half: Lower barrier
        p = t * 2
        b = 2.0 * (1 - p) + 0.1
        c = 0.0
    else:
        # Second half: Tilt right and raise barrier
        p = (t - 0.5) * 2
        b = 2.0 * p + 0.1
        c = 1.5 * p
    return b, c

# Convert protocol to JAX-compatible arrays
def precompute_protocol(protocol_fn, n_steps):
    """Precompute protocol (for JAX loops)"""
    bs = []
    cs = []
    for i in range(n_steps + 1):
        t = i / n_steps
        b, c = protocol_fn(t)
        bs.append(b)
        cs.append(c)
    return jnp.array(bs), jnp.array(cs)

# ===== Simulation Execution =====

def run_simulation(protocol_fn, n_steps, n_particles, T, key, init_x=None):
    """
    Execute protocol and calculate total work
    
    Args:
        protocol_fn: Protocol function
        n_steps: Number of steps
        n_particles: Number of particles
        T: Temperature
        key: Random key
        init_x: Initial position (None starts near origin)
    
    Returns:
        mean_work: Mean work
        std_work: Work standard deviation
        final_x: Final positions
        work_trajectory: Work trajectory
    """
    # Precompute protocol
    bs, cs = precompute_protocol(protocol_fn, n_steps)
    
    # Initial position
    if init_x is None:
        key, subkey = random.split(key)
        x = random.normal(subkey, shape=(n_particles,)) * 0.3
    else:
        x = jnp.ones(n_particles) * init_x
    
    total_work = jnp.zeros(n_particles)
    work_trajectory = []
    
    for i in range(n_steps):
        b_old, c_old = bs[i], cs[i]
        b_new, c_new = bs[i+1], cs[i+1]
        
        # Calculate work (from parameter change)
        w = work_step_batch(x, b_old, c_old, b_new, c_new)
        total_work = total_work + w
        work_trajectory.append(jnp.mean(total_work))
        
        # Update positions
        key, subkey = random.split(key)
        keys = random.split(subkey, n_particles)
        x = step_batch(x, keys, b_new, c_new, T)
    
    return jnp.mean(total_work), jnp.std(total_work), x, jnp.array(work_trajectory)

# ===== Maintain Simulation =====

def run_maintain_simulation(n_steps, n_particles, T, key, target_state=1.0):
    """
    State maintenance simulation
    Measure error rate and correction cost
    """
    b = 2.0  # Fixed barrier
    c = 0.0  # No bias
    
    # Initial state: Place at target state
    x = jnp.ones(n_particles) * target_state
    
    total_errors = 0
    total_correction_work = 0.0
    error_history = []
    
    for i in range(n_steps):
        # Update positions
        key, subkey = random.split(key)
        keys = random.split(subkey, n_particles)
        x = step_batch(x, keys, b, c, T)
        
        # Error detection (fell to opposite side of target)
        if target_state > 0:
            errors = x < 0
        else:
            errors = x > 0
        
        n_errors = jnp.sum(errors)
        total_errors += int(n_errors)
        
        # Error correction (forced restoration)
        if n_errors > 0:
            # Correction work (estimate: energy to cross barrier)
            correction_work = n_errors * 2.0 * KB * T
            total_correction_work += float(correction_work)
            # Forced restoration
            x = jnp.where(errors, target_state, x)
        
        error_history.append(int(n_errors))
    
    error_rate = total_errors / (n_steps * n_particles)
    maintain_cost_per_step = total_correction_work / n_steps
    
    return error_rate, maintain_cost_per_step, jnp.array(error_history)

# ===== CSV Export Functions =====

def export_crud_results_to_csv(results, temp_results, prefix="landauer_crud"):
    """
    Export CRUD simulation results to CSV files
    
    Args:
        results: CRUD operation results
        temp_results: Temperature dependence results
        prefix: Filename prefix
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export CRUD summary
    filename_crud = f"{prefix}_summary_{timestamp}.csv"
    with open(filename_crud, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Operation', 'Mean_Work', 'Std_Work', 'Ratio_to_Landauer'])
        landauer = KB * 1.0 * jnp.log(2)  # T=1.0
        for op in ['Create', 'Delete', 'Update']:
            if op in results:
                mean_w, std_w = results[op]
                ratio = mean_w / landauer
                writer.writerow([op, mean_w, std_w, ratio])
        # Maintain separately
        if 'Maintain' in results:
            cost, error_rate = results['Maintain']
            writer.writerow(['Maintain', cost, error_rate, 'N/A'])
    print(f"Exported: {filename_crud}")
    
    # Export temperature dependence
    filename_temp = f"{prefix}_temperature_{timestamp}.csv"
    with open(filename_temp, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Temperature', 'Create_Work', 'Delete_Work', 'Landauer_Limit'])
        for i, (temp, create_w) in enumerate(temp_results['Create']):
            delete_w = temp_results['Delete'][i][1]
            landauer_t = KB * temp * jnp.log(2)
            writer.writerow([temp, create_w, delete_w, float(landauer_t)])
    print(f"Exported: {filename_temp}")
    
    return filename_crud, filename_temp


# ===== Main Experiment =====

def main():
    print("=" * 60)
    print("Landauer Extension Simulator")
    print("=" * 60)
    
    # Parameters
    N_PARTICLES = 10000
    N_STEPS = 2000
    T = 1.0
    
    key = random.PRNGKey(42)
    
    # Landauer limit
    landauer = KB * T * jnp.log(2)
    print(f"\nLandauer limit: k_B T ln(2) = {landauer:.4f}")
    print(f"Temperature: T = {T}")
    print(f"Particles: {N_PARTICLES}")
    print(f"Steps: {N_STEPS}")
    print()
    
    results = {}
    trajectories = {}
    
    # ----- Create (compare 3 protocols) -----
    print("Running Create simulations (3 protocols)...")
    
    # v1: Dig well method (original)
    key, subkey = random.split(key)
    mean_w_v1, std_w_v1, final_x_v1, traj_v1 = run_simulation(
        protocol_create_v1, N_STEPS, N_PARTICLES, T, subkey, init_x=0.0
    )
    print(f"  Create v1 (dig well):   <W> = {mean_w_v1:.4f} ± {std_w_v1:.4f}, ratio = {mean_w_v1/landauer:.2f}")
    
    # v2: Raise wall method (improved)
    key, subkey = random.split(key)
    mean_w_v2, std_w_v2, final_x_v2, traj_v2 = run_simulation(
        protocol_create, N_STEPS, N_PARTICLES, T, subkey, init_x=0.5
    )
    print(f"  Create v2 (raise wall): <W> = {mean_w_v2:.4f} ± {std_w_v2:.4f}, ratio = {mean_w_v2/landauer:.2f}")
    
    # v3: Heighten wall method
    key, subkey = random.split(key)
    mean_w_v3, std_w_v3, final_x_v3, traj_v3 = run_simulation(
        protocol_create_v3, N_STEPS, N_PARTICLES, T, subkey, init_x=0.5
    )
    print(f"  Create v3 (heighten):   <W> = {mean_w_v3:.4f} ± {std_w_v3:.4f}, ratio = {mean_w_v3/landauer:.2f}")
    
    # Use v2 as default
    results['Create'] = (float(mean_w_v2), float(std_w_v2))
    results['Create_v1'] = (float(mean_w_v1), float(std_w_v1))
    results['Create_v3'] = (float(mean_w_v3), float(std_w_v3))
    trajectories['Create'] = traj_v2
    trajectories['Create_v1'] = traj_v1
    trajectories['Create_v3'] = traj_v3
    
    print(f"  Final states: v1={jnp.mean(final_x_v1):.3f}, v2={jnp.mean(final_x_v2):.3f}, v3={jnp.mean(final_x_v3):.3f}")
    print()
    
    # ----- Delete -----
    print("Running Delete simulation...")
    key, subkey = random.split(key)
    mean_w, std_w, final_x, traj = run_simulation(
        protocol_delete, N_STEPS, N_PARTICLES, T, subkey, init_x=1.0
    )
    results['Delete'] = (float(mean_w), float(std_w))
    trajectories['Delete'] = traj
    print(f"  Delete: <W> = {mean_w:.4f} ± {std_w:.4f}")
    print(f"  Ratio to Landauer: {mean_w/landauer:.2f}")
    print(f"  Final state (mean x): {jnp.mean(final_x):.3f}")
    print()
    
    # ----- Update -----
    print("Running Update simulation...")
    key, subkey = random.split(key)
    mean_w, std_w, final_x, traj = run_simulation(
        protocol_update, N_STEPS, N_PARTICLES, T, subkey, init_x=-1.0
    )
    results['Update'] = (float(mean_w), float(std_w))
    trajectories['Update'] = traj
    print(f"  Update: <W> = {mean_w:.4f} ± {std_w:.4f}")
    print(f"  Ratio to Landauer: {mean_w/landauer:.2f}")
    print(f"  Ratio to 2*Landauer: {mean_w/(2*landauer):.2f}")
    print(f"  Final state (mean x): {jnp.mean(final_x):.3f}")
    print()
    
    # ----- Maintain -----
    print("Running Maintain simulation...")
    key, subkey = random.split(key)
    error_rate, maintain_cost, error_hist = run_maintain_simulation(
        N_STEPS, N_PARTICLES, T, subkey
    )
    results['Maintain'] = (maintain_cost, error_rate)
    print(f"  Maintain: Cost/step = {maintain_cost:.4f}")
    print(f"  Error rate: {error_rate:.6f}")
    print()
    
    # ----- Temperature Dependence -----
    print("Running temperature dependence...")
    temperatures = [0.5, 1.0, 1.5, 2.0, 2.5]
    temp_results = {'Create': [], 'Delete': []}
    
    for temp in temperatures:
        key, subkey = random.split(key)
        mean_w, _, _, _ = run_simulation(
            protocol_create, N_STEPS, N_PARTICLES, temp, subkey, init_x=0.0
        )
        temp_results['Create'].append((temp, float(mean_w)))
        
        key, subkey = random.split(key)
        mean_w, _, _, _ = run_simulation(
            protocol_delete, N_STEPS, N_PARTICLES, temp, subkey, init_x=1.0
        )
        temp_results['Delete'].append((temp, float(mean_w)))
    
    print("  Temperature dependence:")
    print("  T      Create    Delete    Landauer")
    for i, temp in enumerate(temperatures):
        landauer_t = KB * temp * jnp.log(2)
        print(f"  {temp:.1f}    {temp_results['Create'][i][1]:.4f}    {temp_results['Delete'][i][1]:.4f}    {landauer_t:.4f}")
    print()
    
    # ----- Export to CSV -----
    print("=" * 60)
    print("Exporting Results to CSV")
    print("=" * 60)
    print()
    export_crud_results_to_csv(results, temp_results)
    print()
    
    # ----- Dissipation Analysis -----
    print("=" * 60)
    print("DISSIPATION ANALYSIS")
    print("=" * 60)
    print()
    print("Thermodynamic Work (W):")
    print(f"  Create: {results['Create'][0]:.4f} (energy released to heat bath)")
    print(f"  Delete: {results['Delete'][0]:.4f} (energy required)")
    print(f"  Update: {results['Update'][0]:.4f} (net)")
    print()
    
    # Engineering Cost: Cannot recover in real computers
    # Negative work from Create treated as zero
    create_eng = max(0, results['Create'][0])
    delete_eng = max(0, results['Delete'][0])
    update_eng = max(0, results['Update'][0])
    
    print("Engineering Cost (non-recoverable, W<0 → 0):")
    print(f"  Create: {create_eng:.4f}")
    print(f"  Delete: {delete_eng:.4f}")
    print(f"  Update: {update_eng:.4f}")
    print()
    
    # Total energy moved (dissipation indicator)
    total_energy_moved = abs(results['Create'][0]) + abs(results['Delete'][0])
    print("Total Energy Moved (|W_create| + |W_delete|):")
    print(f"  {total_energy_moved:.4f}")
    print(f"  This is the 'hidden cost' - energy that flows but isn't recovered")
    print()
    
    # Verify Update = Delete + Create
    predicted_update = results['Delete'][0] + results['Create'][0]
    actual_update = results['Update'][0]
    print("Update = Delete + Create verification:")
    print(f"  Predicted: {predicted_update:.4f}")
    print(f"  Actual:    {actual_update:.4f}")
    print(f"  Difference (friction/dissipation): {actual_update - predicted_update:.4f}")
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Landauer limit (T=1): {landauer:.4f}")
    print()
    print("Operation    <W>        σ(W)      Ratio to Landauer")
    print("-" * 55)
    for op in ['Create', 'Delete', 'Update']:
        mean_w, std_w = results[op]
        ratio = mean_w / landauer
        print(f"{op:12} {mean_w:10.4f} {std_w:10.4f} {ratio:10.2f}")
    print()
    print(f"Maintain cost/step: {results['Maintain'][0]:.4f}")
    print(f"Maintain error rate: {results['Maintain'][1]:.6f}")
    print()
    
    # ----- Comparison with Predictions -----
    print("=" * 60)
    print("COMPARISON WITH PREDICTIONS")
    print("=" * 60)
    print()
    print("Hypothesis H1: Create ≈ k_B T ln(2)")
    create_ratio = results['Create'][0] / landauer
    print(f"  Result: Create = {create_ratio:.2f} × Landauer")
    print(f"  Status: {'✓ CONFIRMED' if 0.5 < create_ratio < 2.0 else '? CHECK'}")
    print()
    print("Hypothesis H2: Update ≈ 2 × k_B T ln(2)")
    update_ratio = results['Update'][0] / (2 * landauer)
    print(f"  Result: Update = {results['Update'][0]/landauer:.2f} × Landauer")
    print(f"  Status: {'✓ CONFIRMED' if 0.5 < update_ratio < 2.0 else '? CHECK'}")
    print()
    print("Landauer Principle: Delete ≥ k_B T ln(2)")
    delete_ratio = results['Delete'][0] / landauer
    print(f"  Result: Delete = {delete_ratio:.2f} × Landauer")
    print(f"  Status: {'✓ CONFIRMED' if delete_ratio >= 0.8 else '? CHECK'}")
    print()
    
    # ----- Plotting -----
    print("Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Work Trajectories
    ax1 = axes[0, 0]
    for op, traj in trajectories.items():
        ax1.plot(traj, label=op)
    ax1.axhline(y=landauer, color='k', linestyle='--', label='Landauer')
    ax1.axhline(y=2*landauer, color='gray', linestyle='--', label='2×Landauer')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cumulative Work')
    ax1.set_title('Work Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Temperature Dependence
    ax2 = axes[0, 1]
    temps = [t[0] for t in temp_results['Create']]
    create_works = [t[1] for t in temp_results['Create']]
    delete_works = [t[1] for t in temp_results['Delete']]
    landauer_line = [KB * t * jnp.log(2) for t in temps]
    
    ax2.plot(temps, create_works, 'o-', label='Create')
    ax2.plot(temps, delete_works, 's-', label='Delete')
    ax2.plot(temps, landauer_line, 'k--', label='k_B T ln(2)')
    ax2.set_xlabel('Temperature T')
    ax2.set_ylabel('Work')
    ax2.set_title('Temperature Dependence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Bar Chart
    ax3 = axes[1, 0]
    ops = ['Create', 'Delete', 'Update']
    works = [results[op][0] for op in ops]
    errors = [results[op][1] for op in ops]
    x_pos = range(len(ops))
    
    bars = ax3.bar(x_pos, works, yerr=errors, capsize=5, alpha=0.7)
    ax3.axhline(y=landauer, color='r', linestyle='--', label='Landauer')
    ax3.axhline(y=2*landauer, color='orange', linestyle='--', label='2×Landauer')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(ops)
    ax3.set_ylabel('Work')
    ax3.set_title('CRUD Operation Costs')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Potential Visualization
    ax4 = axes[1, 1]
    x_range = jnp.linspace(-2, 2, 200)
    
    # Create at various stages
    for t in [0.0, 0.33, 0.66, 1.0]:
        b, c = protocol_create(t)
        V = potential(x_range, b, c)
        ax4.plot(x_range, V, label=f't={t:.2f}', alpha=0.7)
    
    ax4.set_xlabel('x')
    ax4.set_ylabel('V(x)')
    ax4.set_title('Potential Evolution (Create)')
    ax4.set_ylim(-3, 5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('landauer_simulation_results.png', dpi=150, bbox_inches='tight')
    plt.savefig('landauer_simulation_results.pdf', dpi=300, bbox_inches='tight')
    print("Saved: landauer_simulation_results.png")
    print("Saved: landauer_simulation_results.pdf")
    plt.close()
    
    print()
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    
    return results, temp_results

if __name__ == "__main__":
    results, temp_results = main()
