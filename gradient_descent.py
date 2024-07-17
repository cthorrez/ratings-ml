import math
import numpy as np
import matplotlib.pyplot as plt
from riix.models.autograd_rating_system import AutogradRatingSystem
from riix.eval import evaluate
from data_utils import load_dataset
from jax.scipy.stats import norm, cauchy, logistic, laplace
import gc
import jax
import multiprocessing as mp

def evaluate_single_point(args):
    dataset, test_mask, cdf, scale, lr = args
    rs = AutogradRatingSystem(
        dataset.competitors,
        cdf=cdf,
        initial_rating=0.0,
        learning_rate=lr,
        scale=scale,
        update_method='batched'
    )
    metrics = evaluate(rs, dataset, test_mask)
    log_loss = metrics['log_loss']
    print(f'scale: {scale}, lr: {lr}, log loss: {log_loss}')
    return log_loss

def hyperparameter_sweep(dataset, test_mask, cdf, num_points=8):
    scales = np.logspace(-1, math.log10(200.0), num_points)
    learning_rates = np.logspace(-2, 2, num_points)
    
    args_list = [(dataset, test_mask, cdf, scale, lr) for scale in scales for lr in learning_rates]
    
    with mp.Pool(processes=8) as pool:
        results = pool.map(evaluate_single_point, args_list)
    
    results = np.array(results).reshape(num_points, num_points)
    
    return results, scales, learning_rates

def plot_heatmap(results, scales, learning_rates, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(results, cmap='coolwarm', aspect='auto', origin='lower', extent=[np.log10(learning_rates[0]), np.log10(learning_rates[-1]), np.log10(scales[0]), np.log10(scales[-1])])
    plt.colorbar(label='Log Loss')
    plt.xlabel('Log10(Learning Rate)')
    plt.ylabel('Scale')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_heatmap_2.png")
    plt.close()

def main():
    # dataset, test_mask = load_dataset('tetris', test_start_date='2023-06-30')
    dataset, test_mask = load_dataset('league_of_legends', test_start_date='2023-06-30')

    print(f"Dataset size: {len(dataset)}, Test set size: {test_mask.sum()}")

    distributions = [
        ('Logistic', logistic.cdf),
        ('Cauchy', cauchy.cdf),
        ('Gaussian', norm.cdf),
        ('Laplace', laplace.cdf)
    ]

    for dist_name, cdf in distributions:
        print(f"Running hyperparameter sweep for {dist_name} distribution...")
        results, scales, learning_rates = hyperparameter_sweep(dataset, test_mask, cdf)
        plot_heatmap(results, scales, learning_rates, f"{dist_name} Distribution Hyperparameter Sweep")
        print(f"Heatmap saved as {dist_name.lower()}_distribution_hyperparameter_sweep.png")
        
        # Clear memory after each distribution
        gc.collect()
        jax.clear_caches()

if __name__ == '__main__':
    main()