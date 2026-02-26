import pandas as pd
import numpy as np

class DataPreprocessor:

    def __init__(self, games_info, bgg_reviews):
        self.games_info = games_info
        self.bgg_reviews = bgg_reviews
        self.idx2game = None
    #merging datasets with inner join on 'ID' column, optionally only with selected features from games_info dataset, and returning merged dataframe
    def merge_datasets(self, featires_to_add=None):

        if featires_to_add is not None:
            self.games_info = self.games_info[featires_to_add + ['ID']]
        merged_df = pd.merge(self.bgg_reviews, self.games_info, on='ID', how='inner')
        return merged_df

    # Discretize column using qcut into n_bins, optionally only for top percent of data, and optionally drop original column
    def discretize_column(self, column, n_bins=10, percent=None, drop_original=False):
        df = self.games_info.copy()

        #cuting only top percent of data
        if percent is not None:
            df.sort_values(by=column, ascending=False, inplace=True)
            n = int(percent * len(df))
            top_rows = df.iloc[:n]
        else:
            top_rows = df

        # qcut diskretizacija
        bins = pd.qcut(top_rows[column], q=n_bins, labels=False, duplicates='drop')
        top_rows = top_rows.assign(**{f"{column}_bin": bins})

        # Optionally delete original column
        if drop_original:
            top_rows.drop(columns=[column], inplace=True)

        # Update internal games_info and return the trimmed dataset
        self.games_info = top_rows
        return None

    # convert dataframe to list of tuples (user_idx, item_idx, feature1_idx, ..., rating) and return also number of users, items, features and mapping dicts for users, items and features
    def data_to_tuple(self, df, user_col='user', item_col='ID',
                  feature_col=None, rating_col='rating'):

        users = df[user_col].unique()
        items = df[item_col].unique()

        user2idx = {u: i for i, u in enumerate(users)}
        # game to index and index to game dictionaries
        game2idx = {g: i for i, g in enumerate(items)}
        idx2game = {i: g for g, i in game2idx.items()}
        self.idx2game = idx2game

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

    # Split data entries into train and test sets based on test_size and seed, return train and test entries
    def train_test_split(self, data_entries, test_size=0.2, seed=42):
        np.random.seed(seed)

        data_entries = list(data_entries)  # osiguraj da je lista
        np.random.shuffle(data_entries)    # shuffle radi i na listama

        n_test = int(len(data_entries) * test_size)

        test_entries = data_entries[:n_test]
        train_entries = data_entries[n_test:]

        """
        Keep only the records in the test set where the user and item are already in the train set.
        This is important because the model cannot give recommendations for users or games it hasn't seen during training."""

        train_users = set()
        train_items = set()

        # izvučemo sve user i item indekse iz traina
        for entry in train_entries:
            u_idx, m_idx, *_ = entry
            train_users.add(u_idx)
            train_items.add(m_idx)

        # filtriramo test
        filtered_test = [
            entry for entry in test_entries
            if entry[0] in train_users and entry[1] in train_items
        ]


        return train_entries, filtered_test

    def get_game_name(self, m_idx):
        if self.idx2game is None:
            return "Unknown"
        game_id = self.idx2game[m_idx]
        row = self.games_info.loc[self.games_info["ID"] == game_id]
        if not row.empty:
            return row["name"].values[0]
        return "Unknown"

    def top_games_for_user_name_wraper(self, top ,user_idx):
        pretty_results = []
        for m_idx, score in top:
            name = self.get_game_name(m_idx)
            pretty_results.append((name, score))

        return pretty_results