import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from policy import PolicyNet
from mountaincar import estimate_J

NN_ARCH_BEST = (4, 4) 
P_POPULATION_SIZE = 50  
K_ELITE_SIZE    = 5   
SIGMA_EXPLORATION = 0.2 
ALPHA_STEP_SIZE   = 0.15

def _eval_theta(args):
    arch, theta, num_episodes = args
    p_local = PolicyNet(neurons_per_layer=arch)
    return estimate_J(p_local, theta, num_episodes=num_episodes)

def evolution_strategies(
    policy_architecture: tuple,
    num_iterations: int,
    P: int,
    K: int,
    sigma: float,
    alpha: float,
    seed: int = None
):
    if seed is not None:
        np.random.seed(seed)

    p = PolicyNet(neurons_per_layer=policy_architecture)
    theta_t = p.get_policy_parameters()
    parameters = theta_t.size
    J_mean = []
    
    print(f"  Params: P={P}, K={K}, sigma={sigma}, alpha={alpha}, NN={policy_architecture}")

    ctx = mp.get_context("spawn")
    processes = max(1, mp.cpu_count())
    with ctx.Pool(processes=processes) as pool:
        for t in range(num_iterations):
            J = estimate_J(p, theta_t, num_episodes=15)
            J_mean.append(J)

            epsilons = np.random.standard_normal((P, parameters)).astype(np.float32)
            thetas = theta_t + sigma * epsilons

            args = [(policy_architecture, thetas[i], 15) for i in range(P)]
            chunksize = max(1, P // (processes * 4))
            J_list = pool.map(_eval_theta, args, chunksize)

            perturbs = list(zip(J_list, epsilons, thetas))
            perturbs.sort(key=lambda x: x[0], reverse=True)
            K_res = perturbs[:K]
            gradsum = np.zeros(parameters, dtype=np.float32)
            for J_k, e_k, _ in K_res:
                gradsum += (J_k - J) * e_k
            grad = (1.0 / sigma) * (1.0 / K) * gradsum
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-8:
                grad = grad / grad_norm
            theta_t = theta_t + alpha * grad
        
    print(f"Final Return = {J_mean[-1]:.2f}\n")
    p.set_policy_parameters(theta_t)
    best_t = p.get_policy_parameters()
    return J_mean, best_t

def plot_graph(results_dict, title, num_runs):
    plt.figure(figsize=(10, 6))
    
    for label, all_returns in results_dict.items():
        returns_array = np.array(all_returns)
        mean_J = np.mean(returns_array, axis=0)
        std_J = np.std(returns_array, axis=0)       
        iters = np.arange(len(mean_J))
        plt.plot(iters, mean_J, label=f'{label} (N={num_runs} runs)')

        plt.fill_between(iters, mean_J - std_J, mean_J + std_J, alpha=0.1)

    plt.axhline(-120, color='r', linestyle='--', label='Optimal')
    
    plt.title(title)
    plt.xlabel('ES Iteration')
    plt.ylabel(f'Mean Return')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.ylim(min(-1000, plt.ylim()[0]), max(0, plt.ylim()[1])) 
    plt.show()

def _compute_stats(returns_array: np.ndarray):
    mean_J = np.mean(returns_array, axis=0)
    std_J = np.std(returns_array, axis=0)
    final_mean = float(mean_J[-1])
    final_std = float(std_J[-1])
    best_iter = int(np.argmax(mean_J))
    best_mean = float(mean_J[best_iter])
    best_std = float(std_J[best_iter])
    stability = float(np.mean(std_J)) 
    return {
        'final_mean': final_mean,
        'final_std': final_std,
        'best_iter': best_iter,
        'best_mean': best_mean,
        'best_std': best_std,
        'stability': stability,
    }

def run_exp(configs: dict, num_runs: int, plot_title: str):
    all_results = {}
    for name, params in configs.items():
        
        all_run_returns = []
        print(f"\n{name} for {num_runs} runs")
        
        for run_id in range(num_runs):
            returns, _ = evolution_strategies(
                policy_architecture=params['nn'],
                num_iterations=30,
                P=params['P'],
                K=K_ELITE_SIZE,
                sigma=params['sigma'],
                alpha=params['alpha'],
                seed=run_id 
            )
            all_run_returns.append(returns)
            
        all_results[name] = all_run_returns
        
    plot_graph(all_results, plot_title, num_runs)
    
    os.makedirs('outputs', exist_ok=True)
    safe_title = plot_title.replace(' ', '_').replace('/', '-').replace(':', '-')
    csv_path = os.path.join('outputs', f"stats_{safe_title}.csv")
    with open(csv_path, 'w') as f:
        f.write('config,final_mean,final_std,best_iter,best_mean,best_std,stability\n')
        best_label = None
        best_final_mean = -1e9
        for label, all_returns in all_results.items():
            returns_array = np.array(all_returns)
            stats = _compute_stats(returns_array)
            print(f"[Stats] {label}: final={stats['final_mean']:.2f}±{stats['final_std']:.2f}, "
                  f"best@{stats['best_iter']}={stats['best_mean']:.2f}±{stats['best_std']:.2f}, "
                  f"stability(avg std)={stats['stability']:.2f}")
            f.write(
                f"{label},{stats['final_mean']:.6f},{stats['final_std']:.6f},{stats['best_iter']},"
                f"{stats['best_mean']:.6f},{stats['best_std']:.6f},{stats['stability']:.6f}\n"
            )
            if stats['final_mean'] > best_final_mean:
                best_final_mean = stats['final_mean']
                best_label = label
    print(f"Saved stats to: {csv_path}")
    return all_results, best_label


if __name__ == "__main__":

    print(f"\nProblem 1 Results\n")

    q1_configs = {
        "1. (P50, S0.2, A0.15, NN4x4)": {
            'P': 50, 'sigma': 0.2, 'alpha': 0.15, 'nn': NN_ARCH_BEST
        },
        "2. (P30, S0.2, A0.15, NN4x4)": {
            'P': 30, 'sigma': 0.2, 'alpha': 0.15, 'nn': NN_ARCH_BEST
        },
        "3. (P50, S0.1, A0.15, NN4x4)": {
            'P': 50, 'sigma': 0.1, 'alpha': 0.15, 'nn': NN_ARCH_BEST
        },
        "4. (P50, S0.2, A0.5, NN4x4)": {
            'P': 50, 'sigma': 0.2, 'alpha': 0.5, 'nn': NN_ARCH_BEST
        },
        "5. (P50, S0.2, A0.15, NN2x2)": {
            'P': 50, 'sigma': 0.2, 'alpha': 0.15, 'nn': (2, 2)
        },
    }
    
    q1_results, q1_best_label = run_exp(
        configs=q1_configs,
        num_runs=5,
        plot_title="Hyperparameter Search averaged over 5 Runs"
    )

    print(f"\nProblem 2 Results\n")
    
    q2_config = {
        "Best Hyperparameters": q1_configs[q1_best_label]
    }
    
    run_exp(
        configs=q2_config,
        num_runs=15,
        plot_title=f"Best Policy Performance averaged over 15 runs"
    )
    
