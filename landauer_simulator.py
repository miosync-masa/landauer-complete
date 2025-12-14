"""
Landauer Extension Simulator
============================
CRUD操作の熱力学的コストをシミュレーション

理論予測：
  - Create: ~ k_B T ln(2)
  - Delete: ~ k_B T ln(2) (Landauer)
  - Update: ~ 2 * k_B T ln(2)
  - Maintain: エラー率に依存

Author: Masamichi Iizumi & Tamaki
Date: December 14, 2025
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import matplotlib.pyplot as plt

print("JAX devices:", jax.devices())

# ===== 物理定数 =====
KB = 1.0      # ボルツマン定数（単位系）
GAMMA = 1.0   # 摩擦係数
DT = 0.001    # 時間刻み

# ===== ポテンシャル =====
# V(x) = x^4 - b*x^2 - c*x
# b: 障壁の高さ（正で二重井戸）
# c: 傾き（操作バイアス）

@jit
def potential(x, b, c):
    return x**4 - b * x**2 - c * x

@jit
def force(x, b, c):
    """力 = -dV/dx（解析的）"""
    return -4*x**3 + 2*b*x + c

# ===== 1ステップの時間発展 =====
@jit
def step_single(x, key, b, c, T):
    """ランジュバン方程式の1ステップ（Euler-Maruyama）"""
    f = force(x, b, c)
    noise = random.normal(key)
    x_new = x + (f / GAMMA) * DT + jnp.sqrt(2 * KB * T * DT / GAMMA) * noise
    return x_new

# バッチ版
@jit
def step_batch(x_batch, keys, b, c, T):
    """複数粒子を並列更新"""
    return vmap(lambda x, k: step_single(x, k, b, c, T))(x_batch, keys)

# ===== 仕事の計算 =====
@jit
def work_step(x, b_old, c_old, b_new, c_new):
    """パラメータ変化による仕事 W = V_new - V_old"""
    return potential(x, b_new, c_new) - potential(x, b_old, c_old)

work_step_batch = vmap(work_step, in_axes=(0, None, None, None, None))

# ===== プロトコル定義 =====

def protocol_create_v1(t):
    """Create v1（元のバージョン）: 井戸を掘る方式"""
    b = 2.0 * t       # 障壁を徐々に形成
    c = 1.0 * t       # 右へ傾ける
    return b, c

def protocol_create(t):
    """Create v2（Gemini提案）: 壁を立てる方式
    
    最初から二重井戸があり、壁を高くして閉じ込める
    これにより「位置エネルギーを下げる」ではなく
    「自由度を制限する」操作になる
    """
    # Phase 1 (t < 0.3): 傾けて右井戸に誘導
    # Phase 2 (0.3 <= t < 0.7): 傾きを維持して安定化  
    # Phase 3 (t >= 0.7): 壁を高くして閉じ込め
    
    # 連続関数で表現
    b = 2.0 + 1.0 * jnp.maximum(0, (t - 0.7) / 0.3)  # 2 → 3 (t > 0.7で増加)
    c = 0.5 * jnp.minimum(1.0, t / 0.3) * jnp.maximum(0, 1 - jnp.maximum(0, (t - 0.7) / 0.3))
    return b, c

def protocol_create_v3(t):
    """Create v3: より明確な「壁を立てる」方式
    
    最初は浅い井戸、徐々に深くする（壁を高くする）
    位置は変えず、閉じ込めを強くする
    """
    # 最初から右側に配置、壁を高くしていく
    b = 1.0 + 2.0 * t   # 1 → 3 （障壁を高くする）
    c = 0.3             # 軽い右バイアス（固定）
    return b, c

def protocol_delete(t):
    """Delete: 確定状態 → 左井戸（状態0）へ消去（ランダウアー）"""
    b = 2.0 * (1 - t) + 0.1  # 障壁を下げる
    c = -1.5 * t             # 左へ強く傾ける
    return b, c

def protocol_update(t):
    """Update: 左井戸 → 右井戸（状態変更）"""
    if t < 0.5:
        # 前半: 障壁を下げる
        p = t * 2
        b = 2.0 * (1 - p) + 0.1
        c = 0.0
    else:
        # 後半: 右へ傾けて障壁を上げる
        p = (t - 0.5) * 2
        b = 2.0 * p + 0.1
        c = 1.5 * p
    return b, c

# プロトコルをJAX互換の配列に変換
def precompute_protocol(protocol_fn, n_steps):
    """プロトコルを事前計算（JAXのループ用）"""
    bs = []
    cs = []
    for i in range(n_steps + 1):
        t = i / n_steps
        b, c = protocol_fn(t)
        bs.append(b)
        cs.append(c)
    return jnp.array(bs), jnp.array(cs)

# ===== シミュレーション実行 =====

def run_simulation(protocol_fn, n_steps, n_particles, T, key, init_x=None):
    """
    プロトコルを実行し、総仕事を計算
    
    Args:
        protocol_fn: プロトコル関数
        n_steps: ステップ数
        n_particles: 粒子数
        T: 温度
        key: 乱数キー
        init_x: 初期位置（Noneなら原点付近からスタート）
    
    Returns:
        mean_work: 平均仕事
        std_work: 仕事の標準偏差
        final_x: 最終位置
        work_trajectory: 仕事の軌跡
    """
    # プロトコルを事前計算
    bs, cs = precompute_protocol(protocol_fn, n_steps)
    
    # 初期位置
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
        
        # 仕事を計算（パラメータ変化による）
        w = work_step_batch(x, b_old, c_old, b_new, c_new)
        total_work = total_work + w
        work_trajectory.append(jnp.mean(total_work))
        
        # 位置を更新
        key, subkey = random.split(key)
        keys = random.split(subkey, n_particles)
        x = step_batch(x, keys, b_new, c_new, T)
    
    return jnp.mean(total_work), jnp.std(total_work), x, jnp.array(work_trajectory)

# ===== Maintain シミュレーション =====

def run_maintain_simulation(n_steps, n_particles, T, key, target_state=1.0):
    """
    状態維持のシミュレーション
    エラー率と補正コストを測定
    """
    b = 2.0  # 固定障壁
    c = 0.0  # バイアスなし
    
    # 初期状態：ターゲット状態に配置
    x = jnp.ones(n_particles) * target_state
    
    total_errors = 0
    total_correction_work = 0.0
    error_history = []
    
    for i in range(n_steps):
        # 位置を更新
        key, subkey = random.split(key)
        keys = random.split(subkey, n_particles)
        x = step_batch(x, keys, b, c, T)
        
        # エラー検出（ターゲットと反対側に落ちた）
        if target_state > 0:
            errors = x < 0
        else:
            errors = x > 0
        
        n_errors = jnp.sum(errors)
        total_errors += int(n_errors)
        
        # エラー補正（強制復帰）
        if n_errors > 0:
            # 補正の仕事（概算：障壁を超えるエネルギー）
            correction_work = n_errors * 2.0 * KB * T
            total_correction_work += float(correction_work)
            # 強制復帰
            x = jnp.where(errors, target_state, x)
        
        error_history.append(int(n_errors))
    
    error_rate = total_errors / (n_steps * n_particles)
    maintain_cost_per_step = total_correction_work / n_steps
    
    return error_rate, maintain_cost_per_step, jnp.array(error_history)

# ===== メイン実験 =====

def main():
    print("=" * 60)
    print("Landauer Extension Simulator")
    print("=" * 60)
    
    # パラメータ
    N_PARTICLES = 10000
    N_STEPS = 2000
    T = 1.0
    
    key = random.PRNGKey(42)
    
    # ランダウアー限界
    landauer = KB * T * jnp.log(2)
    print(f"\nLandauer limit: k_B T ln(2) = {landauer:.4f}")
    print(f"Temperature: T = {T}")
    print(f"Particles: {N_PARTICLES}")
    print(f"Steps: {N_STEPS}")
    print()
    
    results = {}
    trajectories = {}
    
    # ----- Create (3つのプロトコル比較) -----
    print("Running Create simulations (3 protocols)...")
    
    # v1: 井戸を掘る方式（元）
    key, subkey = random.split(key)
    mean_w_v1, std_w_v1, final_x_v1, traj_v1 = run_simulation(
        protocol_create_v1, N_STEPS, N_PARTICLES, T, subkey, init_x=0.0
    )
    print(f"  Create v1 (dig well):   <W> = {mean_w_v1:.4f} ± {std_w_v1:.4f}, ratio = {mean_w_v1/landauer:.2f}")
    
    # v2: 壁を立てる方式（Gemini提案）
    key, subkey = random.split(key)
    mean_w_v2, std_w_v2, final_x_v2, traj_v2 = run_simulation(
        protocol_create, N_STEPS, N_PARTICLES, T, subkey, init_x=0.5
    )
    print(f"  Create v2 (raise wall): <W> = {mean_w_v2:.4f} ± {std_w_v2:.4f}, ratio = {mean_w_v2/landauer:.2f}")
    
    # v3: 壁を高くする方式
    key, subkey = random.split(key)
    mean_w_v3, std_w_v3, final_x_v3, traj_v3 = run_simulation(
        protocol_create_v3, N_STEPS, N_PARTICLES, T, subkey, init_x=0.5
    )
    print(f"  Create v3 (heighten):   <W> = {mean_w_v3:.4f} ± {std_w_v3:.4f}, ratio = {mean_w_v3/landauer:.2f}")
    
    # デフォルトはv2を使用
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
    
    # ----- 温度依存性 -----
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
    
    # ----- 散逸仕事とEngineering Cost -----
    print("=" * 60)
    print("DISSIPATION ANALYSIS (Gemini's insight)")
    print("=" * 60)
    print()
    print("Thermodynamic Work (W):")
    print(f"  Create: {results['Create'][0]:.4f} (energy released to heat bath)")
    print(f"  Delete: {results['Delete'][0]:.4f} (energy required)")
    print(f"  Update: {results['Update'][0]:.4f} (net)")
    print()
    
    # Engineering Cost: 現実のコンピュータでは回収できない
    # Create の負の仕事はゼロとして扱う
    create_eng = max(0, results['Create'][0])
    delete_eng = max(0, results['Delete'][0])
    update_eng = max(0, results['Update'][0])
    
    print("Engineering Cost (non-recoverable, W<0 → 0):")
    print(f"  Create: {create_eng:.4f}")
    print(f"  Delete: {delete_eng:.4f}")
    print(f"  Update: {update_eng:.4f}")
    print()
    
    # 全移動エネルギー（散逸の指標）
    total_energy_moved = abs(results['Create'][0]) + abs(results['Delete'][0])
    print("Total Energy Moved (|W_create| + |W_delete|):")
    print(f"  {total_energy_moved:.4f}")
    print(f"  This is the 'hidden cost' - energy that flows but isn't recovered")
    print()
    
    # Update = Delete + Create の検証
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
    
    # ----- 理論予測との比較 -----
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
    
    # ----- プロット -----
    print("Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 仕事の軌跡
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
    
    # 2. 温度依存性
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
    
    # 3. バーチャート
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
    
    # 4. ポテンシャル可視化
    ax4 = axes[1, 1]
    x_range = jnp.linspace(-2, 2, 200)
    
    # Create の各段階
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
    plt.savefig('landauer_simulation_results.png', dpi=150)
    print("Saved: landauer_simulation_results.png")
    plt.close()
    
    print()
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    
    return results, temp_results

if __name__ == "__main__":
    results, temp_results = main()
