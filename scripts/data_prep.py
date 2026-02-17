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

