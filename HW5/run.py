import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from environment import Environment
from ValueIteration import ValueIteration

from rl_algorithms import (
    TDConfiguration,
    SarsaAgent,
    QLearningAgent,
    TDLearning,
    format_policy,
    format_value,
    greedypol_q,
    max_norm,
)

matplotlib.use("Agg")


def save_table(table, path: Path):
    with path.open("w") as f:
        for row in table:
            f.write("\t".join(row) + "\n")


def plot_curve(x, y, xlabel, ylabel, title, path: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, linewidth=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def td_learning(env: Environment, vals_opt, pol_opt, results: Path):
    tdc = TDConfiguration(alpha=0.18, delta=1e-4, min_episodes=500, max_episodes=6000, max_steps=250)
    runs,eps = [],[]

    for seed in range(50):
        evaluator = TDLearning(env, pol_opt, tdc)
        vals, episodes = evaluator.run(seed)
        runs.append(vals)
        eps.append(episodes)

    vals_avg = {}
    for s in env.all_states():
        vals_avg[s] = float(np.mean([run[s] for run in runs]))

    stats = {
        "alpha": tdc.alpha,
        "delta": tdc.delta,
        "min_episodes": tdc.min_episodes,
        "max_episodes": tdc.max_episodes,
        "mean_episodes": float(np.mean(eps)),
        "std_episodes": float(np.std(eps)),
        "max_norm_vs_optimal": max_norm(vals_avg, vals_opt, set(env.furniture)),
    }

    (results / "td_summary.json").write_text(json.dumps(stats, indent=2))
    save_table(format_value(vals_avg, env), results / "td_average_value.txt")


def averagestats_runs(stats):
    steps = np.vstack([run["steps_history"] for run in stats])
    mse = np.vstack([run["mse_history"] for run in stats])
    q_tables = np.stack([run["q_values"] for run in stats])
    return steps.mean(axis=0), mse.mean(axis=0), q_tables.mean(axis=0)


def sarsa(env: Environment, vals_opt, results: Path):
    config = {
        "alpha": 0.25,
        "epsilon": 0.30,
        "epsilon_decay": 0.995,
        "min_epsilon": 0.02,
        "q_init": 1.0,
        "max_steps": 300,
    }
    runct, epsct = 20, 400
    sarsa_stats = []
    for seed in range(runct):
        qa = SarsaAgent(env, **config)
        stats = qa.train(epsct, seed, vals_opt)
        stats["q_values"] = qa.q_values.copy()
        sarsa_stats.append(stats)

    avg_steps, avg_mse, avg_q = averagestats_runs(sarsa_stats)
    eps_dir = np.arange(1, epsct + 1)
    plot_curve(
        avg_steps,
        eps_dir,
        "Total No. of actions",
        "Episodes ",
        "SARSA Learning Curve",
        results / "sarsa_actions_vs_episodes.png",
    )
    plot_curve(
        eps_dir,
        avg_mse,
        "Episodes",
        "MSE",
        "SARSA MSE Curve",
        results / "sarsa_mse_curve.png",
    )
    pol = greedypol_q(avg_q, env)
    save_table(format_policy(pol, env), results / "sarsa_greedy_policy.txt")
    curve_data = {
        "episodes": eps_dir.tolist(),
        "avg_actions": avg_steps.tolist(),
        "avg_mse": avg_mse.tolist(),
    }
    (results / "sarsa_curves.json").write_text(json.dumps(curve_data, indent=2))

    summary = {
        "design_choices": {
            "alpha": config["alpha"],
            "q_init": config["q_init"],
            "epsilon_strategy": "ε-greedy with exponential decay",
            "epsilon_start": config["epsilon"],
            "epsilon_decay": config["epsilon_decay"],
            "epsilon_min": config["min_epsilon"],
            "episodes": epsct,
            "max_steps_per_episode": config["max_steps"],
        }
    }
    (results / "sarsa_summary.json").write_text(json.dumps(summary, indent=2))


def q_learning(env: Environment, vals_opt, results: Path):
    config = {
        "alpha": 0.20,
        "epsilon": 0.25,
        "epsilon_decay": 0.996,
        "min_epsilon": 0.02,
        "q_init": 1.0,
        "max_steps": 300,
    }
    runct, epsct = 20, 400
    q_stats = []
    for seed in range(runct):
        qa = QLearningAgent(env, **config)
        stats = qa.train(epsct, seed, vals_opt)
        stats["q_values"] = qa.q_values.copy()
        q_stats.append(stats)

    avg_steps, avg_mse, avg_q = averagestats_runs(q_stats)
    eps_dir = np.arange(1, epsct + 1)
    plot_curve(
        avg_steps,
        eps_dir,
        "Total No. of actions",
        "Episodes",
        "Q-Learning Learning Curve",
        results / "q_learning_actions_vs_episodes.png",
    )
    plot_curve(
        eps_dir,
        avg_mse,
        "Episodes",
        "MSE",
        "Q-Learning MSE Curve",
        results / "q_learning_mse_curve.png",
    )
    pol = greedypol_q(avg_q, env)
    save_table(format_policy(pol, env), results / "q_learning_greedy_policy.txt")
    curve_data = {
        "episodes": eps_dir.tolist(),
        "avg_actions": avg_steps.tolist(),
        "avg_mse": avg_mse.tolist(),
    }
    (results / "q_learning_curves.json").write_text(json.dumps(curve_data, indent=2))

    summary = {
        "design_choices": {
            "alpha": config["alpha"],
            "q_init": config["q_init"],
            "epsilon_strategy": "ε-greedy with exponential decay",
            "epsilon_start": config["epsilon"],
            "epsilon_decay": config["epsilon_decay"],
            "epsilon_min": config["min_epsilon"],
            "episodes": epsct,
            "max_steps_per_episode": config["max_steps"],
        }
    }
    (results / "q_learning_summary.json").write_text(json.dumps(summary, indent=2))


def main():
    env = Environment(gamma=0.925)
    vi = ValueIteration(env)
    vals_opt, pol_opt, x = vi.run()
    results = Path("results")
    results.mkdir(exist_ok=True)

    save_table(format_value(vals_opt, env), results / "optimal_value_table.txt")
    save_table(format_policy(pol_opt, env), results / "optimal_policy_table.txt")

    td_learning(env, vals_opt, pol_opt, results)
    sarsa(env, vals_opt, results)
    q_learning(env, vals_opt, results)


if __name__ == "__main__":
    main()

