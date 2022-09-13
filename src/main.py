# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from abc import ABC

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch

from src.ncf import NCF


def import_data():
    np.random.seed(123)

    data = pd.read_csv('../data/ratings.csv')

    # subset the data
    rand_user_ids = np.random.choice(data['userId'].unique(),
                                     size=int(len(data['userId'].unique()) * 0.2),
                                     replace=False)

    data = data.loc[data['userId'].isin(rand_user_ids)]

    return data


def split_data(reviews):
    reviews['rank_latest'] = reviews.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)

    # split the train/test split by the latest rating
    train_reviews = reviews[reviews['rank_latest'] != 1]
    test_reviews = reviews[reviews['rank_latest'] == 1]

    # drop columns that we no longer need
    train_reviews = train_reviews[['userId', 'movieId', 'rating']]
    test_reviews = test_reviews[['userId', 'movieId', 'rating']]

    # set rating to label if the user has watched the movie
    train_reviews.loc[:, 'rating'] = 1

    return train_reviews, test_reviews


def evaluate(reviews, test_reviews):
    # User-item pairs for testing
    test_user_item_set = set(zip(test_reviews['userId'], test_reviews['movieId']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = reviews.groupby('userId')['movieId'].apply(list).to_dict()

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


def get_recommendations(reviews, test_reviews, user_id):
    movies = pd.read_csv('../data/movies.csv')
    test_user_item_set = set(zip(test_reviews['userId'], test_reviews['movieId']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = reviews.groupby('userId')['movieId'].apply(list).to_dict()

    for (u, i) in test_user_item_set:
        recommendations = ''
        if u == int(user_id):
            interacted_items = user_interacted_items[u]
            not_interacted_items = set(all_movieIds) - set(interacted_items)
            selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
            test_items = selected_not_interacted + [i]
            predicted_labels = np.squeeze(model(torch.tensor([u] * 100),
                                                torch.tensor(test_items)).detach().numpy())
            top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
            print(top10_items)
            for it in top10_items:
                recommendations += movies.loc[movies['movieId'] == it, 'title'].item() + '\n'
            print("Tested is:", movies.loc[movies['movieId'] == i, 'title'].item())
            print("We recommended you:\n", recommendations)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    subdata = import_data()

    train_data, test_data = split_data(subdata)

    num_users = subdata['userId'].max() + 1
    num_items = subdata['movieId'].max() + 1
    all_movieIds = subdata['movieId'].unique()

    checkpoint = torch.load('checkpoints/epoch=0-step=2520-v1.ckpt', map_location=lambda storage, loc: storage)
    hyper_params = checkpoint["hyper_parameters"]

    print("Train the model(1) or use the already trained one(2)?")
    choice = input()

    if choice == '1':
        model = NCF(torch.tensor(num_users).to(torch.int64), torch.tensor(num_items).to(torch.int64), train_data,
                    all_movieIds)
        trainer = pl.Trainer(max_epochs=3, reload_dataloaders_every_epoch=True,
                             progress_bar_refresh_rate=50, logger=False)
        trainer.fit(model)
    else:
        model = NCF.load_from_checkpoint('checkpoints/epoch=0-step=2520-v1.ckpt',
                                         num_users=hyper_params["num_users"], num_items=hyper_params["num_items"],
                                         ratings=hyper_params["ratings"],
                                         all_movie_ids=hyper_params["all_movie_ids"])

    while True:
        print("Overall testing(1) or single user testing(2)?")
        test = input()
        if test == '':
            break
        elif test == '1':
            evaluate(subdata, test_data)
        else:
            print("User Id:")
            x = input()
            get_recommendations(subdata, test_data, x)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
