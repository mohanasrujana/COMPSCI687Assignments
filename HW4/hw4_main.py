import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except (ImportError, AttributeError) as e:
    HAS_MATPLOTLIB = False
    print(f"Warning: matplotlib not available ({e}). Plotting will be skipped.")
    print("To fix: pip install 'numpy<2' or upgrade matplotlib")

from monte_carlo import Environment, ValueIteration, arrow_formatting, MonteCarlo, MonteCarloControl


def print_value_function(env: Environment, values: dict, title: str = "Value Function"):
    """Print value function in grid format"""
    print(f"\n{title}")
    for r in range(env.rows):
        row_vals = []
        for c in range(env.cols):
            state = (r, c)
            if env.is_blocked(state):
                row_vals.append("0.0000")
            else:
                row_vals.append(f"{values[state]:.4f}")
        print("\t".join(row_vals))


def print_policy(env: Environment, policy: dict, title: str = "Policy"):
    """Print policy in grid format"""
    print(f"\n{title}")
    for r in range(env.rows):
        row_pol = []
        for c in range(env.cols):
            state = (r, c)
            if env.is_blocked(state):
                row_pol.append("")
            elif env.is_terminal(state):
                row_pol.append("G")
            else:
                row_pol.append(arrow_formatting(policy[state]))
        print("\t".join(row_pol))


def compute_max_norm(v1: dict, v2: dict, env: Environment) -> float:
    """Compute max-norm between two value functions"""
    max_diff = 0.0
    for state in env.all_states():
        if not env.is_blocked(state) and not env.is_terminal(state):
            max_diff = max(max_diff, abs(v1[state] - v2[state]))
    return max_diff


def compute_mse(v1: dict, v2: dict, env: Environment) -> float:
    """Compute mean squared error between two value functions"""
    errors = []
    for state in env.all_states():
        if not env.is_blocked(state) and not env.is_terminal(state):
            errors.append((v1[state] - v2[state]) ** 2)
    return np.mean(errors) if errors else 0.0


def question_1a():
    """Question 1a: First-Visit Monte Carlo"""
    print("=" * 80)
    print("Question 1a: First-Visit Monte Carlo")
    print("=" * 80)
    
    # Create environment and get optimal policy
    env = Environment(gamma=0.925)
    vi = ValueIteration(env)
    true_values, optimal_policy, _ = vi.run()
    
    # Run First-Visit Monte Carlo
    mc_first = MonteCarlo(env, optimal_policy, first_visit=True)
    estimated_values, iterations = mc_first.run(
        max_iterations=100000,
        convergence_threshold=0.1,
        true_values=true_values
    )
    
    print(f"\nConverged after {iterations} iterations")
    print(f"Max-norm error: {compute_max_norm(estimated_values, true_values, env):.4f}")
    print_value_function(env, estimated_values, "Estimated Value Function (First-Visit MC)")
    print_value_function(env, true_values, "True Value Function (from Value Iteration)")
    
    return estimated_values, iterations, true_values


def question_1b():
    """Question 1b: Every-Visit Monte Carlo"""
    print("\n" + "=" * 80)
    print("Question 1b: Every-Visit Monte Carlo")
    print("=" * 80)
    
    # Create environment and get optimal policy
    env = Environment(gamma=0.925)
    vi = ValueIteration(env)
    true_values, optimal_policy, _ = vi.run()
    
    # Run Every-Visit Monte Carlo
    mc_every = MonteCarlo(env, optimal_policy, first_visit=False)
    estimated_values, iterations = mc_every.run(
        max_iterations=100000,
        convergence_threshold=0.1,
        true_values=true_values
    )
    
    print(f"\nConverged after {iterations} iterations")
    print(f"Max-norm error: {compute_max_norm(estimated_values, true_values, env):.4f}")
    print_value_function(env, estimated_values, "Estimated Value Function (Every-Visit MC)")
    print_value_function(env, true_values, "True Value Function (from Value Iteration)")
    
    return estimated_values, iterations, true_values


def question_1c(first_iterations: int, every_iterations: int):
    """Question 1c: Compare First-Visit and Every-Visit"""
    print("\n" + "=" * 80)
    print("Question 1c: Comparison of First-Visit and Every-Visit Monte Carlo")
    print("=" * 80)
    
    print(f"\nFirst-Visit Monte Carlo required {first_iterations} iterations")
    print(f"Every-Visit Monte Carlo required {every_iterations} iterations")
    
    if first_iterations < every_iterations:
        print("\nFirst-Visit Monte Carlo converged faster (fewer iterations needed).")
        print("This is expected because First-Visit uses each state's first occurrence")
        print("in a trajectory, which can lead to faster convergence in some cases.")
    elif every_iterations < first_iterations:
        print("\nEvery-Visit Monte Carlo converged faster (fewer iterations needed).")
        print("This can happen because Every-Visit uses all occurrences of states,")
        print("potentially providing more data per trajectory.")
    else:
        print("\nBoth methods required the same number of iterations.")
    
    return first_iterations, every_iterations


def question_2a():
    """Question 2a: Monte Carlo Control with different epsilon values"""
    print("\n" + "=" * 80)
    print("Question 2a: Monte Carlo Control with ε-soft policies")
    print("=" * 80)
    
    env = Environment(gamma=0.925)
    vi = ValueIteration(env)
    true_values, _, _ = vi.run()
    
    epsilon_values = [0.2, 0.1, 0.05]
    results = {}
    
    for epsilon in epsilon_values:
        print(f"\n{'=' * 80}")
        print(f"Running with ε = {epsilon}")
        print(f"{'=' * 80}")
        
        mc_control = MonteCarloControl(env, epsilon=epsilon)
        estimated_values = mc_control.run(num_iterations=10000)
        
        mse = compute_mse(estimated_values, true_values, env)
        max_norm = compute_max_norm(estimated_values, true_values, env)
        
        print(f"MSE: {mse:.6f}")
        print(f"Max-norm: {max_norm:.4f}")
        print_value_function(env, estimated_values, f"Value Function (ε = {epsilon})")
        
        results[epsilon] = {
            'values': estimated_values,
            'mse': mse,
            'max_norm': max_norm
        }
    
    return results, true_values


def question_2b(results: dict, true_values: dict):
    """Question 2b: Learning curves for different epsilon values"""
    print("\n" + "=" * 80)
    print("Question 2b: Learning Curves")
    print("=" * 80)
    
    env = Environment(gamma=0.925)
    epsilon_values = [0.2, 0.1, 0.05]
    num_iterations = 10000
    eval_interval = 250
    
    learning_curves = {}
    
    for epsilon in epsilon_values:
        print(f"\nGenerating learning curve for ε = {epsilon}...")
        mc_control = MonteCarloControl(env, epsilon=epsilon)
        
        mse_history = []
        iteration_points = []
        
        # Manually run iterations to track learning curve
        for iteration in range(num_iterations):
            # Sample trajectory and update
            trajectory = []
            state = mc_control._get_initial_state()
            
            while not env.is_terminal(state):
                action = mc_control._sample_action(state)
                next_states = list(env.transitions[state][action].keys())
                probs = list(env.transitions[state][action].values())
                next_state_idx = np.random.choice(len(next_states), p=probs)
                next_state = next_states[next_state_idx]
                reward = env.rewards[state][action][next_state]
                trajectory.append((state, action, reward))
                state = next_state
            
            mc_control.update(trajectory)
            
            # Evaluate every 250 iterations
            if (iteration + 1) % eval_interval == 0:
                estimated_values = mc_control.get_value_function()
                mse = compute_mse(estimated_values, true_values, env)
                mse_history.append(mse)
                iteration_points.append(iteration + 1)
        
        learning_curves[epsilon] = {
            'iterations': iteration_points,
            'mse': mse_history
        }
    
    # Plot learning curves
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(12, 8))
        for epsilon in epsilon_values:
            plt.plot(learning_curves[epsilon]['iterations'], 
                    learning_curves[epsilon]['mse'],
                    label=f'ε = {epsilon}', linewidth=2)
        
        plt.xlabel('Number of Iterations', fontsize=12)
        plt.ylabel('Mean Squared Error', fontsize=12)
        plt.title('Learning Curves: MSE vs Iterations for Different ε Values', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('HW_figs/learning_curves_fixed_epsilon.png', dpi=300, bbox_inches='tight')
        print("\nLearning curve saved to HW_figs/learning_curves_fixed_epsilon.png")
        plt.close()
    else:
        print("\nSkipping plot generation (matplotlib not available)")
        print("Learning curve data available in learning_curves dictionary")
    
    return learning_curves


def question_2c():
    """Question 2c: Monte Carlo Control with epsilon decay schedule"""
    print("\n" + "=" * 80)
    print("Question 2c: Monte Carlo Control with ε Decay Schedule")
    print("=" * 80)
    
    env = Environment(gamma=0.925)
    vi = ValueIteration(env)
    true_values, _, _ = vi.run()
    
    num_iterations = 10000
    initial_epsilon = 1.0
    final_epsilon = 0.05
    decay_interval = 500
    decay_amount = 0.05
    
    def epsilon_schedule(iteration: int) -> float:
        """Decay epsilon by 0.05 every 500 iterations"""
        decay_steps = iteration // decay_interval
        epsilon = initial_epsilon - decay_steps * decay_amount
        return max(epsilon, final_epsilon)
    
    print(f"\nEpsilon schedule: starts at {initial_epsilon}, decays by {decay_amount} every {decay_interval} iterations")
    print(f"Final epsilon: {final_epsilon}")
    
    mc_control = MonteCarloControl(env, epsilon=initial_epsilon)
    
    mse_history = []
    iteration_points = []
    eval_interval = 250
    
    for iteration in range(num_iterations):
        # Update epsilon
        mc_control.epsilon = epsilon_schedule(iteration)
        
        # Sample trajectory and update
        trajectory = []
        state = mc_control._get_initial_state()
        
        while not env.is_terminal(state):
            action = mc_control._sample_action(state)
            next_states = list(env.transitions[state][action].keys())
            probs = list(env.transitions[state][action].values())
            next_state_idx = np.random.choice(len(next_states), p=probs)
            next_state = next_states[next_state_idx]
            reward = env.rewards[state][action][next_state]
            trajectory.append((state, action, reward))
            state = next_state
        
        mc_control.update(trajectory)
        
        # Evaluate every 250 iterations
        if (iteration + 1) % eval_interval == 0:
            estimated_values = mc_control.get_value_function()
            mse = compute_mse(estimated_values, true_values, env)
            mse_history.append(mse)
            iteration_points.append(iteration + 1)
    
    final_values = mc_control.get_value_function()
    final_mse = compute_mse(final_values, true_values, env)
    final_max_norm = compute_max_norm(final_values, true_values, env)
    
    print(f"\nFinal MSE: {final_mse:.6f}")
    print(f"Final Max-norm: {final_max_norm:.4f}")
    print_value_function(env, final_values, "Final Value Function (with ε decay)")
    
    # Plot learning curve
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(12, 8))
        plt.plot(iteration_points, mse_history, label='ε decay schedule', linewidth=2, color='red')
        plt.xlabel('Number of Iterations', fontsize=12)
        plt.ylabel('Mean Squared Error', fontsize=12)
        plt.title('Learning Curve: MSE vs Iterations with ε Decay Schedule', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('HW_figs/learning_curve_decay.png', dpi=300, bbox_inches='tight')
        print("\nLearning curve saved to HW_figs/learning_curve_decay.png")
        plt.close()
    else:
        print("\nSkipping plot generation (matplotlib not available)")
        print(f"Final MSE: {final_mse:.6f}")
    
    return final_values, final_mse, final_max_norm, iteration_points, mse_history


def question_2d(results_2a: dict, results_2c: dict):
    """Question 2d: Compare different variants"""
    print("\n" + "=" * 80)
    print("Question 2d: Comparison of Different Variants")
    print("=" * 80)
    
    print("\nFinal MSE values:")
    print("-" * 80)
    for epsilon, result in results_2a.items():
        print(f"Fixed ε = {epsilon}: MSE = {result['mse']:.6f}, Max-norm = {result['max_norm']:.4f}")
    
    print(f"\nε decay schedule: MSE = {results_2c['mse']:.6f}, Max-norm = {results_2c['max_norm']:.4f}")
    
    # Find best variant
    best_fixed = min(results_2a.items(), key=lambda x: x[1]['mse'])
    best_epsilon = best_fixed[0]
    best_fixed_mse = best_fixed[1]['mse']
    
    if results_2c['mse'] < best_fixed_mse:
        print(f"\n✓ ε decay schedule performed best with MSE = {results_2c['mse']:.6f}")
        print("\nInterpretation:")
        print("The decay schedule allows the algorithm to explore more initially (high ε)")
        print("and then exploit more as it learns (low ε). This balance between exploration")
        print("and exploitation leads to better convergence to the optimal value function.")
    else:
        print(f"\n✓ Fixed ε = {best_epsilon} performed best with MSE = {best_fixed_mse:.6f}")
        print("\nInterpretation:")
        print(f"A fixed epsilon value of {best_epsilon} provided the best balance between")
        print("exploration and exploitation for this problem. The decay schedule may have")
        print("decayed too quickly or too slowly to match this performance.")


def main():
    """Run all questions"""
    print("\n" + "=" * 80)
    print("COMPSCI 687 Homework 4 - Monte Carlo Methods")
    print("=" * 80)
    
    # Question 1
    print("\n" + "#" * 80)
    print("QUESTION 1: Monte Carlo Policy Evaluation")
    print("#" * 80)
    
    # 1a: First-Visit
    first_values, first_iterations, true_values = question_1a()
    
    # 1b: Every-Visit
    every_values, every_iterations, _ = question_1b()
    
    # 1c: Comparison
    question_1c(first_iterations, every_iterations)
    
    # Question 2
    print("\n" + "#" * 80)
    print("QUESTION 2: Monte Carlo Control with ε-soft policies")
    print("#" * 80)
    
    # 2a: Different epsilon values
    results_2a, true_values_2 = question_2a()
    
    # 2b: Learning curves
    learning_curves = question_2b(results_2a, true_values_2)
    
    # 2c: Epsilon decay
    final_values_2c, final_mse_2c, final_max_norm_2c, iter_points, mse_history = question_2c()
    results_2c = {
        'values': final_values_2c,
        'mse': final_mse_2c,
        'max_norm': final_max_norm_2c
    }
    
    # 2d: Comparison
    question_2d(results_2a, results_2c)
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

