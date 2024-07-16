import numpy as np
import matplotlib.pyplot as plt
from riix.models.autograd_rating_system import AutogradRatingSystem
from riix.eval import evaluate
from data_utils import load_dataset
from jax.scipy.stats import norm, cauchy, logistic, laplace

def hyperparameter_sweep(dataset, test_mask, cdf, num_points=20):
    # Create logarithmically spaced values for scale and learning rate
    scales = np.logspace(-3, 1, num_points)
    learning_rates = np.logspace(-3, 1, num_points)
    
    # Initialize the results matrix
    results = np.zeros((num_points, num_points))
    
    for i, scale in enumerate(scales):
        for j, lr in enumerate(learning_rates):
            rs = AutogradRatingSystem(
                dataset.competitors,
                cdf=cdf,
                initial_rating=0.0,
                learning_rate=lr,
                scale=scale,
                update_method='batched'
            )
            metrics = evaluate(rs, dataset, test_mask)
            results[i, j] = metrics['log_loss']
    
    return results, scales, learning_rates

def plot_heatmap(results, scales, learning_rates, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(results, cmap='coolwarm', aspect='auto', origin='lower', extent=[np.log10(learning_rates[0]), np.log10(learning_rates[-1]), np.log10(scales[0]), np.log10(scales[-1])])
    plt.colorbar(label='Log Loss')
    plt.xlabel('Log10(Learning Rate)')
    plt.ylabel('Log10(Scale)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_heatmap.png")
    plt.close()

def main():
    dataset, test_mask = load_dataset('league_of_legends', test_start_date='2023-03-31')
    print(f"Dataset size: {len(dataset)}, Test set size: {test_mask.sum()}")

    # List of distributions to test
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

if __name__ == '__main__':
    main()
