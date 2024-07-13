import datasets
from riix.utils.data_utils import MatchupDataset

def load_dataset(game, test_start_date):
    df = datasets.load_dataset(
        'EsportsBench/EsportsBench',
        split=game
    ).to_pandas()
    dataset = MatchupDataset(
        df=df,
        competitor_cols=['competitor_1', 'competitor_2'],
        datetime_col='date',
        outcome_col='outcome'
    )
    test_mask = df['date'] > test_start_date
    return dataset, test_mask