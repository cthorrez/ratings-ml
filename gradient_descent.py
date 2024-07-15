import math
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from riix.models.autograd_rating_system import AutogradRatingSystem
from riix.models.elo import Elo
from riix.eval import evaluate
from data_utils import load_dataset

@jax.jit
def logistic_predict(ratings, matchups):
    matchup_ratings = ratings[matchups]
    neg_rating_diffs = matchup_ratings[:, 1] - matchup_ratings[:, 0]
    probs = jax.nn.sigmoid(neg_rating_diffs)
    return probs

@jax.jit
def logistic_likelihood(ratings, matchups, outcomes):
    matchup_ratings = ratings[matchups]
    neg_rating_diffs = matchup_ratings[:, 1] - matchup_ratings[:, 0]
    probs = jax.nn.sigmoid(neg_rating_diffs)
    return -jnp.log((outcomes * probs) + ((1.0 - outcomes) * (1.0 - probs))).sum()


@jax.jit
def cauchy_predict(ratings, matchups):
    matchup_ratings = ratings[matchups]
    rating_diffs = matchup_ratings[:, 0] - matchup_ratings[:, 1]
    probs = (1.0 / jnp.pi) * jnp.arctan(rating_diffs) + 0.5
    return probs

@jax.jit
def cauchy_likelihood(ratings, matchups, outcomes):
    matchup_ratings = ratings[matchups]
    rating_diffs = matchup_ratings[:, 0] - matchup_ratings[:, 1]
    probs = (1.0 / jnp.pi) * jnp.arctan(rating_diffs) + 0.5
    return -jnp.log((outcomes * probs) + ((1.0 - outcomes) * (1.0 - probs))).sum()

@jax.jit
def gaussian_predict(ratings, matchups):
    matchup_ratings = ratings[matchups]
    rating_diffs = matchup_ratings[:, 0] - matchup_ratings[:, 1]
    probs = norm.cdf(rating_diffs)
    return probs

@jax.jit
def gaussian_likelihood(ratings, matchups, outcomes):
    matchup_ratings = ratings[matchups]
    rating_diffs = matchup_ratings[:, 0] - matchup_ratings[:, 1]
    probs = norm.cdf(rating_diffs)
    return -jnp.log((outcomes * probs) + ((1.0 - outcomes) * (1.0 - probs))).sum()


def main():
    dataset, test_mask = load_dataset('league_of_legends', test_start_date='2023-03-31')
    print(len(dataset), test_mask.sum())

    logistic = AutogradRatingSystem(
        dataset.competitors,
        predict_fn=logistic_predict,
        likelihood_fn=logistic_likelihood,
        initial_rating=0.0,
        learning_rate=0.2,
        update_method='batched'
    )
    logistic_metrics = evaluate(logistic, dataset, test_mask)
    print('logistic')
    print(logistic_metrics)

    cauchy = AutogradRatingSystem(
        dataset.competitors,
        predict_fn=cauchy_predict,
        likelihood_fn=cauchy_likelihood,
        initial_rating=0.0,
        learning_rate=0.2,
        update_method='batched'
    )
    cauchy_metrics = evaluate(cauchy, dataset, test_mask)
    print('cauchy')
    print(cauchy_metrics)

    gaussian = AutogradRatingSystem(
        dataset.competitors,
        predict_fn=gaussian_predict,
        likelihood_fn=gaussian_likelihood,
        initial_rating=0.0,
        learning_rate=0.2,
        update_method='batched'
    )
    gaussian_metrics = evaluate(gaussian, dataset, test_mask)
    print('gaussian')
    print(gaussian_metrics)

    elo = Elo(competitors=dataset.competitors, update_method='batched')
    elo_metrics = evaluate(elo, dataset, test_mask)
    print('elo')
    print(elo_metrics)


if __name__ == '__main__':
    main()