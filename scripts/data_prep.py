import pandas as pd
import numpy as np

class DataPreprocessor:

    def __init__(self, games_info, bgg_reviews):
        self.games_info = games_info
        self.bgg_reviews = bgg_reviews

    def merge_datasets(self, featires_to_add=None):

        if featires_to_add is not None:
            self.games_info = self.games_info[featires_to_add + ['ID']]
        merged_df = pd.merge(self.bgg_reviews, self.games_info, on='ID', how='inner')
        return merged_df

    def discretize_column(self, df, column, n_bins=10, percent=None, drop_original=False):
        df = df.copy()

        if percent is not None:
            df.sort_values(by=column, ascending=False, inplace=True)
            n = int(percent * len(df))
            top_rows = df.iloc[:n]
        else:
            top_rows = df

        # qcut diskretizacija
        bins = pd.qcut(top_rows[column], q=n_bins, labels=False, duplicates='drop')
        top_rows = top_rows.assign(**{f"{column}_bin": bins})

        # Opcionalno brisanje originalne kolone
        if drop_original:
            top_rows.drop(columns=[column], inplace=True)

        # Vrati samo podrezani dataset
        return top_rows

    def data_to_tuple(self, df, user_col='user', item_col='ID',
                  feature_col=None, rating_col='rating'):

        users = df[user_col].unique()
        items = df[item_col].unique()

        user2idx = {u: i for i, u in enumerate(users)}
        game2idx = {g: i for i, g in enumerate(items)}

        feature2idx = {}
        for col in feature_col:
            values = df[col].unique()
            feature2idx[col] = {v: i for i, v in enumerate(values)}

        user_idx = df[user_col].map(user2idx).values
        game_idx = df[item_col].map(game2idx).values
        rating = df[rating_col].values

        feature_indices = [
            df[col].map(feature2idx[col]).values
            for col in feature_col
        ]

        data_entries = list(zip(
            user_idx,
            game_idx,
            *feature_indices,
            rating
        ))

        num_users = len(users)
        num_games = len(items)
        num_features = {col: len(feature2idx[col]) for col in feature_col}

        return data_entries, num_users, num_games, num_features, user2idx, game2idx, feature2idx

    def train_test_split(self, data_entries, test_size=0.2, seed=42):
        np.random.seed(seed)
        data_entries = np.array(data_entries)
        np.random.shuffle(data_entries)

        n_test = int(len(data_entries) * test_size)
        test_entries = data_entries[:n_test]
        train_entries = data_entries[n_test:]

        return train_entries.tolist(), test_entries.tolist()

