"""
Conway's Game of Life density Experiments
Generates CSV data for thesis analysis
"""
import numpy as np
import pandas as pd

GRID_SIZE   = 256                                 # grid dimensions
ITERATIONS  = 5000                                # simulation steps
SAMPLE_EVERY = 50                                 # sampling interval
DENSITIES   = [i/100 for i in range(0, 101, 5)]   # 0% to 100%
TRIALS      = 10                                  # repetitions per density
GLOBAL_SEED = 34                                  # reproducibility seed


def neighbor_sum(grid: np.ndarray) -> np.ndarray:       # count live neighbors
    s = np.zeros_like(grid, dtype=np.uint8)

    def roll_and_mask(arr, shift, axis):                # shift grid and mask edges
        out = np.roll(arr, shift=shift, axis=axis)
        if axis == 0:  # vertical shifts
            if shift > 0:
                out[:shift, :] = 0
            elif shift < 0:
                out[shift:, :] = 0
        else:          # horizontal shifts
            if shift > 0:
                out[:, :shift] = 0
            elif shift < 0:
                out[:, shift:] = 0
        return out

    s += roll_and_mask(grid,  1, 0)     # 8 neighbors
    s += roll_and_mask(grid, -1, 0)
    s += roll_and_mask(grid,  1, 1)
    s += roll_and_mask(grid, -1, 1)
    s += roll_and_mask(roll_and_mask(grid,  1, 0),  1, 1)
    s += roll_and_mask(roll_and_mask(grid,  1, 0), -1, 1)
    s += roll_and_mask(roll_and_mask(grid, -1, 0),  1, 1)
    s += roll_and_mask(roll_and_mask(grid, -1, 0), -1, 1)
    return s

def step(grid: np.ndarray) -> np.ndarray:      # apply GoL rules
    nbrs = neighbor_sum(grid)
    survive = (grid == 1) & ((nbrs == 2) | (nbrs == 3))
    born    = (grid == 0) & (nbrs == 3)
    return (survive | born).astype(np.uint8)

def _trial_seed(p: float, trial: int) -> int:      # unique seed per trial
    return GLOBAL_SEED * 10_000 + int(round(p * 100)) * 100 + trial

def run_experiment(iterations=ITERATIONS, trials=TRIALS,    # main simulation loop
                   densities=DENSITIES, grid_size=GRID_SIZE,
                   sample_every=SAMPLE_EVERY):                        
    rows = []  
    # time series for each density
    n_samples = iterations // sample_every + 1  # include t=0
    timeseries_mean = {p: np.zeros(n_samples, dtype=np.float64) for p in densities}

    for p in densities:
        for trial in range(trials):
            seed = _trial_seed(p, trial)
            rng = np.random.default_rng(seed)

            # init grid by density p
            g = (rng.random((grid_size, grid_size)) < p).astype(np.uint8)

            # time series sampling
            sample_idx = 0
            for it in range(iterations + 1): 
                if it % sample_every == 0:
                    timeseries_mean[p][sample_idx] += g.mean()
                    sample_idx += 1
                if it < iterations:
                    g = step(g)

            final_alive = g.sum()
            rows.append({
                "density": p,
                "trial": trial,
                "seed": seed,
                "final_alive_fraction": final_alive / (grid_size * grid_size),
                "survived": int(final_alive > 0),
            })
            print(f"p={p:.2f} trial={trial} final_alive={final_alive}")

    # average time series across trials
    for p in densities:
        timeseries_mean[p] /= trials

    df_final = pd.DataFrame(rows)

    iters = [i for i in range(0, iterations + 1, sample_every)]
    ts_rows = []
    for p in densities:
        for idx, it in enumerate(iters):
            ts_rows.append({
                "iter": it,
                "density": p,
                "mean_alive_fraction": timeseries_mean[p][idx]
            })
    df_ts = pd.DataFrame(ts_rows)

    return df_final, df_ts

def aggregate(df_final: pd.DataFrame):   # calculate statistics
    grouped = df_final.groupby("density", as_index=False).agg(
        mean_final_alive=("final_alive_fraction", "mean"),
        std_final_alive=("final_alive_fraction", "std"),
        survival_prob=("survived", "mean"),
    )

    return grouped

def main():
    df_final, df_ts = run_experiment()
    df_final.to_csv("results.csv", index=False)
    df_ts.to_csv("timeseries_aggregated.csv", index=False)
    agg = aggregate(df_final)
    agg.to_csv("results_aggregated.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    main()
