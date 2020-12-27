import pandas as pd
import numpy as np
from torch.utils.data import Sampler
from sklearn.cluster import AgglomerativeClustering


class IndexRandomSampler(Sampler):
    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self.rng = np.random.default_rng(seed)

    def shuffle_index(self, df):
        indices = df.index.unique()
        return df.loc[self.rng.permutation(indices)]


class SubjectRandomSampler(IndexRandomSampler):
    '''RandomSampler that keeps episodes of patients together.'''

    def __init__(self, data_source, seed=0):
        super().__init__(data_source, seed)
        self.df = self.data_source.file_df.reset_index().reset_index()

    def __iter__(self):
        df = self.df.set_index('subj')
        df = self.shuffle_index(df)

        return iter(df['index'].tolist())

    def __len__(self):
        return len(self.data_source)


class AgeSubjectRandomSampler(IndexRandomSampler):
    '''RandomSampler that keeps episodes of patients of similar age together.'''

    def __init__(self, data_source, age_bins=20, seed=0):
        super().__init__(data_source, seed)
        self.age_bins = age_bins
        self.df = pd.merge(self.data_source.file_df.reset_index().reset_index(),
                           self.data_source.demog_df.reset_index(),
                           on='filename')

    def __iter__(self):
        df = self.df.set_index('subj')
        df = self.shuffle_index(df)

        df.index = pd.cut(df['Age'], self.age_bins, labels=range(self.age_bins))
        df = self.shuffle_index(df)

        return iter(df['index'].tolist())

    def __len__(self):
        return len(self.data_source)


class DiagnoseAgeSubjectRandomSampler(IndexRandomSampler):
    '''RandomSampler that keeps episodes of patients of similar age together.'''

    def __init__(self, data_source, age_bins=20, seed=0):
        super().__init__(data_source, seed)
        self.age_bins = age_bins
        self.df = pd.merge(self.data_source.file_df.reset_index().reset_index(),
                           self.data_source.demog_df.reset_index(),
                           on='filename')
        self.clustering = AgglomerativeClustering(None,
                                                  affinity='manhattan',
                                                  linkage='complete',
                                                  distance_threshold=3,  # Don't allow merging clusters with 2 different diagnoses
                                                  )
        phen_labels = np.array(list(self.df['phenotyping task (labels)'].str.split(';')), dtype=float)
        self.df['phen_cluster'] = self.clustering.fit_predict(phen_labels)

    def __iter__(self):
        df = self.df.set_index('subj')
        df = self.shuffle_index(df)
        df = df.reset_index()

        df['age_group'] = pd.cut(df['Age'], self.age_bins, labels=range(self.age_bins))
        df.set_index('age_group')
        df = self.shuffle_index(df)
        df.reset_index()

        df.index = df['phen_cluster']
        df = self.shuffle_index(df)

        self.sorted_df = df
        return iter(df['index'].tolist())

    def __len__(self):
        return len(self.data_source)
