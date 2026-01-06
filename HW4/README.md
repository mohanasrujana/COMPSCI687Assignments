# Homework 4 - Monte Carlo Methods

This implementation solves Homework 4, which involves implementing Monte Carlo methods for the Cat-vs-Monsters domain.

## Files

- `monte_carlo.py`: Contains the Monte Carlo policy evaluation and control implementations
  - `MonteCarloPolicyEvaluation`: First-Visit and Every-Visit Monte Carlo for policy evaluation
  - `MonteCarloControl`: Monte Carlo control with ε-soft policies

- `hw4_main.py`: Main script that runs all experiments for Questions 1 and 2

- `environment.py`: Cat-vs-Monsters domain environment (from previous homework)

- `ValueIteration.py`: Value Iteration algorithm to get optimal policy and value function (from previous homework)

## Requirements

- Python 3
- numpy
- matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
```

## Running the Code

To run all experiments:

```bash
python3 hw4_main.py
```

This will:
1. Run First-Visit Monte Carlo policy evaluation (Question 1a)
2. Run Every-Visit Monte Carlo policy evaluation (Question 1b)
3. Compare the two methods (Question 1c)
4. Run Monte Carlo control with different ε values (Question 2a)
5. Generate learning curves (Question 2b)
6. Run Monte Carlo control with ε decay schedule (Question 2c)
7. Compare all variants (Question 2d)

## Output

The script will print:
- Value function estimates in grid format
- Number of iterations needed for convergence
- MSE and max-norm errors
- Learning curves saved as PNG files in `HW_figs/`:
  - `learning_curves_fixed_epsilon.png`: Learning curves for fixed ε values
  - `learning_curve_decay.png`: Learning curve for ε decay schedule

## Notes

- The optimal policy and value function are computed using Value Iteration first
- Initial state distribution d₀ is uniform over all non-furniture states
- All value estimates are initialized to zero
- The ε-soft policy update handles ties correctly (multiple optimal actions)

