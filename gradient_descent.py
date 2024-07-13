import math
import jax
import jax.numpy as jnp
from riix.models.autograd_rating_system import AutogradRatingSystem
from riix.eval import evaluate
from data_utils import load_dataset

@jax.jit
def logistic_predict(ratings, matchups, alpha=math.log(10.0) / 400.0):
    matchup_ratings = ratings[matchups]
    neg_rating_diffs = matchup_ratings[:, 1] - matchup_ratings[:, 0]
    probs = jax.nn.sigmoid(alpha * neg_rating_diffs)
    return probs

@jax.jit
def logistic_likelihood(ratings, matchups, outcomes, alpha=math.log(10.0) / 400.0):
    matchup_ratings = ratings[matchups]
    neg_rating_diffs = matchup_ratings[:, 1] - matchup_ratings[:, 0]
    probs = jax.nn.sigmoid(alpha * neg_rating_diffs)
    return -jnp.log((outcomes * probs) + ((1.0 - outcomes) * (1.0 - probs))).sum() / alpha


def main():
    dataset, test_mask = load_dataset('league_of_legends', test_start_date='2023-03-31')
    print(len(dataset), test_mask.sum())

    rating_system = AutogradRatingSystem(dataset.competitors, update_method='batched')

    metrics = evaluate(rating_system, dataset, test_mask)
    print(metrics)


if __name__ == '__main__':
    main()