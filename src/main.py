# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from abc import ABC

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch

from src.ncf import NCF


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def import_data():
    np.random.seed(123)

    ratings = pd.read_csv('../data/ratings.csv')

    rand_user_ids = np.random.choice(ratings['userId'].unique(),
                                     size=int(len(ratings['userId'].unique()) * 0.01),
                                     replace=False)

    ratings = ratings.loc[ratings['userId'].isin(rand_user_ids)]

    return ratings


def split_data(ratings):
    ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)

    train_ratings = ratings[ratings['rank_latest'] != 1]
    test_ratings = ratings[ratings['rank_latest'] == 1]

    # drop columns that we no longer need
    train_ratings = train_ratings[['userId', 'movieId', 'rating']]
    test_ratings = test_ratings[['userId', 'movieId', 'rating']]

    # set rating to label if the user has watched the movie
    train_ratings.loc[:, 'rating'] = 1

    return train_ratings, test_ratings


def evaluate(ratings, test_ratings):
    # User-item pairs for testing
    test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()

    hits = []
    for (u, i) in test_user_item_set:
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_movieIds) - set(interacted_items)
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
        test_items = selected_not_interacted + [i]

        predicted_labels = np.squeeze(model(torch.tensor([u] * 100),
                                            torch.tensor(test_items)).detach().numpy())

        top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]

        if i in top10_items:
            hits.append(1)
        else:
            hits.append(0)

    print("The Hit Ratio @ 10 is {:.2f}".format(np.average(hits)))


def get_recommendations(ratings, test_ratings, user_id):
    movies = pd.read_csv('../data/movies.csv')
    test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()

    for (u, i) in test_user_item_set:
        if u == user_id:
            interacted_items = user_interacted_items[user_id]
            not_interacted_items = set(all_movieIds) - set(interacted_items)
            selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
            test_items = selected_not_interacted + [i]
            predicted_labels = np.squeeze(model(torch.tensor([u] * 100),
                                                torch.tensor(test_items)).detach().numpy())
            top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
            print(top10_items)
            recommendations = ''
            for it in top10_items:
                recommendations += movies.loc[movies['movieId'] == it, 'title'].item() + '\n'
            print("Tested is:", movies.loc[movies['movieId'] == i, 'title'].item())
            print("We recommended you:\n", recommendations)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    data = import_data()
    #print(torch.cuda.is_available(), torch.cuda.device_count())

    train_data, test_data = split_data(data)

    num_users = data['userId'].max() + 1
    num_items = data['movieId'].max() + 1
    all_movieIds = data['movieId'].unique()
    checkpoint = torch.load('checkpoints/epoch=0-step=2520-v1.ckpt', map_location=lambda storage, loc: storage)
    hyper_params = checkpoint["hyper_parameters"]

    model = NCF.load_from_checkpoint('checkpoints/epoch=0-step=2520-v1.ckpt',
                                     num_users=hyper_params["num_users"], num_items=hyper_params["num_items"],
                                     ratings=hyper_params["ratings"],
                                     all_movie_ids=hyper_params["all_movie_ids"])

    if model is None:
        model = NCF(torch.tensor(num_users).to(torch.int64), torch.tensor(num_items).to(torch.int64), train_data,
                all_movieIds)
        trainer = pl.Trainer(max_epochs=1, reload_dataloaders_every_epoch=True,
                         progress_bar_refresh_rate=50, logger=False)
        trainer.fit(model)
    while True:
        print("Overall testing(1) or single user testing(2)?")
        if input() == '1':
            evaluate(data, test_data)
        else:
            print("User Id:")
            x = input()
            get_recommendations(data, test_data, x)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
