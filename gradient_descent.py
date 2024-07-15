import math
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, cauchy, logistic
from riix.models.autograd_rating_system import AutogradRatingSystem
from riix.models.elo import Elo
from riix.eval import evaluate
from data_utils import load_dataset


def main():
    dataset, test_mask = load_dataset('league_of_legends', test_start_date='2023-03-31')
    print(len(dataset), test_mask.sum())

    logistic_rs = AutogradRatingSystem(
        dataset.competitors,
        cdf=logistic.cdf,
        initial_rating=0.0,
        learning_rate=0.2,
        scale=1.0,
        update_method='batched'
    )
    logistic_metrics = evaluate(logistic_rs, dataset, test_mask)
    print('logistic')
    print(logistic_metrics)

    cauchy_rs = AutogradRatingSystem(
        dataset.competitors,
        cdf=cauchy.cdf,
        initial_rating=0.0,
        learning_rate=0.2,
        scale=1.0,
        update_method='batched'
    )
    cauchy_metrics = evaluate(cauchy_rs, dataset, test_mask)
    print('cauchy')
    print(cauchy_metrics)

    gaussian_rs = AutogradRatingSystem(
        dataset.competitors,
        cdf=norm.cdf,
        initial_rating=0.0,
        learning_rate=0.2,
        scale=1.0,
        update_method='batched'
    )
    gaussian_metrics = evaluate(gaussian_rs, dataset, test_mask)
    print('gaussian')
    print(gaussian_metrics)

    elo = Elo(competitors=dataset.competitors, update_method='batched')
    elo_metrics = evaluate(elo, dataset, test_mask)
    print('elo')
    print(elo_metrics)


if __name__ == '__main__':
    main()