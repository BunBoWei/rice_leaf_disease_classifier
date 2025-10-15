import pandas as pd
import os
import argparse

def aggregate_results(args):
    """
    Reads metrics.csv files from all runs, calculates mean and std,
    and prints a summary report.
    """
    all_dfs = []
    results_dir = os.path.join(args.data_root, 'results')

    for i in range(1, args.num_runs + 1):
        run_dir = os.path.join(results_dir, f'run_{i}')
        csv_path = os.path.join(run_dir, 'metrics.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
        else:
            print(f"Warning: Could not find metrics.csv for run {i}. Skipping.")
    
    if not all_dfs:
        print("Error: No result files found to aggregate.")
        return

    full_df = pd.concat(all_dfs)

    # Calculate mean and standard deviation
    summary = full_df.groupby('scenario').agg(
        acc_mean=('accuracy', 'mean'),
        acc_std=('accuracy', 'std'),
        f1_mean=('macro_f1', 'mean'),
        f1_std=('macro_f1', 'std')
    ).reset_index()

    print("\n--- Aggregated Results Summary ---")
    print(f"Averaged over {len(all_dfs)} runs.\n")
    
    for _, row in summary.iterrows():
        print(f"Scenario: {row['scenario']}")
        print(f"  Accuracy:     {row['acc_mean']:.4f} ± {row['acc_std']:.4f}")
        print(f"  Macro F1-Score: {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
        print("-" * 35)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate results from all runs.")
    parser.add_argument('--num_runs', type=int, default=5, help='Total number of runs to aggregate.')
    parser.add_argument('--data_root', type=str, default='./Dhan-Shomadhan', help='Root directory of the dataset.')
    args = parser.parse_args()
    aggregate_results(args)
