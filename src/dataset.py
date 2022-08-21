import torch
from torch.utils.data import Dataset
import numpy as np


class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movie_ids (list): List containing all movieIds

    """

    def __init__(self, ratings, all_movie_ids):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movie_ids)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_movie_ids):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings['userId'], ratings['movieId']))

        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movie_ids)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movie_ids)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)
