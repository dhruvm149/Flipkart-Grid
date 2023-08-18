import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from collections import defaultdict, Counter

from surprise import Reader, Dataset, accuracy
from surprise import SVD, SVDpp, NMF, SlopeOne, NormalPredictor, KNNBaseline, KNNBasic
from surprise import KNNWithMeans, BaselineOnly, CoClustering

from surprise.model_selection import GridSearchCV, KFold
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

original_stdout = sys.stdout


df = pd.read_csv('new_df.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
# Check for duplicated reviews
duplicated = df[df.duplicated(['customer_id', 'product_id'], keep=False)]

pd.concat([df[df.duplicated(['customer_id', 'product_id'], keep=False)].head(2),
           df[df.duplicated(['customer_id', 'product_id'], keep=False)].tail(4)]
         )
df = df[~df.duplicated(['customer_id', 'product_id'], keep=False)].reset_index().drop('index', axis=1)

# Sort df by customer_id and reset index
df = df.set_index('customer_id').sort_values('customer_id')
df = df.reset_index()

np.random.seed(10)
df_ = df.iloc[np.random.choice(df.index, size=10000, replace=False)]
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_[['customer_id', 'product_title', 'star_rating']], reader)

# Cross validate default algorithm
# algo = SVD(n_factors=100,
#             n_epochs=20,
#             biased=True,
#             init_mean=0,
#             init_std_dev=0.1,
#             lr_all=0.005,
#             reg_all=0.02,
#             lr_bu=None,
#             lr_bi=None,
#             lr_pu=None,
#             lr_qi=None,
#             reg_bu=None,
#             reg_bi=None,
#             reg_pu=None,
#             reg_qi=None,
#             random_state=10)
# cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)

# np.random.seed(10)
# df_ = df.iloc[np.random.choice(df.index, size=2000, replace=False)]

# # Shuffle the index of df dataset for random split
# idx = np.array(df_.index, dtype='int')
# np.random.shuffle(idx)

# # Start train-test-split with 80%-20% ratio
# train = df_.loc[idx[:int(0.75*len(idx))],['customer_id', 'product_title', 'star_rating']]
# test = df_.loc[idx[int(0.75*len(idx)):],['customer_id', 'product_title', 'star_rating']]

# # Load trainset and testset into Surprise

# # create a Reader object with the rating_scale from 1 to 5
# # A reader is still needed but only the rating_scale param is required.
# # reader = Reader(rating_scale=(1, 5))

# # # The columns must correspond to user id, item id and ratings (in that order).
# # # Load trainset, note: the columns must correspond to user id, item id and ratings in the exact order
# # data_train = Dataset.load_from_df(train, reader)

# # # Prepare a trainset object out of the training data to feed to .fit() method
# # training = data_train.build_full_trainset()

# # # Load testset
# # data_test = Dataset.load_from_df(test, reader)

# # # Prepare a testset object out of the test data to feed to .test() method
# # testing = data_test.construct_testset(data_test.raw_ratings)


# # Hyperparameter optimization with scikit-surprise SVD algorithm

# # Cross validation to optimize parameters of SVD with bias
# # param_grid = {'n_factors': [50,75,100,150], 
# #                 'n_epochs': [10,20,50,100], 
# #                 'lr_all': [0.002,0.005,0.007],
# #                 'reg_all': [0.01,0.02,0.03,0.04]}

# # svd_gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=KFold(3, random_state=10))
# # svd_gs.fit(data_train) # gridsesarch optimization on the trainset

# # svd = SVD(n_factors=75, n_epochs=100, lr_all=0.002, reg_all=0.01, random_state=10)
# # svd.fit(training)
# # pred_svd = svd.test(testing)

# # Load full dataset and split into train and test set to test hyperparameter on large dataset
# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(df[['customer_id', 'product_title', 'star_rating']], reader)

# trainset, testset = train_test_split(data, test_size=0.25)

# # # Test default algorithm
# # algo = SVD(random_state=10)
# # predictions = algo.fit(trainset).test(testset)

# # # Test tuned algorithm
# # algo_tuned = SVD(n_factors=75, n_epochs=100, lr_all=0.002, reg_all=0.01, random_state=10)
# # predictions_tuned = algo_tuned.fit(trainset).test(testset)

# # Test re-tuned algorithm
# algo_tuned = SVD(n_factors=100, n_epochs=20, lr_all=0.007, reg_all=0.04, random_state=10)
# predictions_tuned = algo_tuned.fit(trainset).test(testset)

# # def get_Iu(uid):
# #     """ return the number of items rated by given user
# #     args: 
# #       uid: the id of the user
# #     returns: 
# #       the number of items rated by the user
# #     """
# #     try:
# #         return len(trainset.ur[trainset.to_inner_uid(uid)])
# #     except ValueError: # user was not part of the trainset
# #         return 0
    
# # def get_Ui(iid):
# #     """ return number of users that have rated given item
# #     args:
# #       iid: the raw id of the item
# #     returns:
# #       the number of users that have rated the item.
# #     """
# #     try: 
# #         return len(trainset.ir[trainset.to_inner_iid(iid)])
# #     except ValueError:
# #         return 0

# # Create dataset for prediction evaluation
# # temp_df = pd.DataFrame(predictions_tuned, columns=['uid', 'iid', 'rui', 'est', 'details'])
# # temp_df['Iu'] = temp_df.uid.apply(get_Iu)
# # temp_df['Ui'] = temp_df.iid.apply(get_Ui)
# # temp_df['err'] = abs(temp_df.est - temp_df.rui)
# # best_predictions = temp_df.sort_values(by='err')[:10]
# # worst_predictions = temp_df.sort_values(by='err')[-10:]

# def get_top_n(predictions, n=10):
#     '''Return the top-N recommendation for each user from a set of predictions.

#     Args:
#         predictions(list of Prediction objects): The list of predictions, as
#             returned by the test method of an algorithm.
#         n(int): The number of recommendation to output for each user. Default
#             is 10.

#     Returns:
#     A dict where keys are user (raw) ids and values are lists of tuples:
#         [(raw item id, rating estimation), ...] of size n.
#     '''

#     # First map the predictions to each user.
#     top_n = defaultdict(list)
#     for uid, iid, true_r, est, _ in predictions:
#         top_n[uid].append((iid, est))

#     # Then sort the predictions for each user and retrieve the k highest ones.
#     for uid, user_ratings in top_n.items():
#         user_ratings.sort(key=lambda x: x[1], reverse=True)
#         top_n[uid] = user_ratings[:n]

#     return top_n

# testset = trainset.build_anti_testset()
# predictions = algo.test(testset)

# top_n = get_top_n(predictions, n=10)

# # Print the recommended items for the first 5 users
# counter = 0
# for uid, user_ratings in top_n.items():
#     if counter == 5:
#         break

    
# # Append recommended items for each user into a dictionary
# top_items = {}
# for uid, user_ratings in top_n.items():
#     top_items[uid] = [iid for (iid, _) in user_ratings]


# recommended_w_ratings = pd.DataFrame.from_dict(top_n, orient='index')
# recommended = pd.DataFrame.from_dict(top_items, orient='index')

# def products_recommended(user):
#     print('The top 10 product recommendations for user {} is:'.format(user))
#     return recommended.loc[user]

# def products_recommended_w_rating(user):
#     number = 0
#     print('The top 10 product recommendations with estimated ratings for user {} is:'.format(user))
#     for rating in recommended_w_ratings.loc[user]:
#         print(number,' ', rating[0], ' : ', rating[1])
#         number += 1

# def precision_recall_at_k(predictions, k=10, threshold=3.5):
#     '''Return precision and recall at k metrics for each user.'''

#     # First map the predictions to each user.
#     user_est_true = defaultdict(list)
#     for uid, _, true_r, est, _ in predictions:
#         user_est_true[uid].append((est, true_r))

#     precisions = dict()
#     recalls = dict()
#     for uid, user_ratings in user_est_true.items():

#         # Sort user ratings by estimated value
#         user_ratings.sort(key=lambda x: x[0], reverse=True)

#         # Number of relevant items
#         n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

#         # Number of recommended items in top k
#         n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

#         # Number of relevant and recommended items in top k
#         n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
#                               for (est, true_r) in user_ratings[:k])

#         # Precision@K: Proportion of recommended items that are relevant
#         precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

#         # Recall@K: Proportion of relevant items that are recommended
#         recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

#     return precisions, recalls

# # Load sampled 10000 ratings dataset
# data = Dataset.load_from_df(df_[['customer_id', 'product_id', 'star_rating']], reader)
# kf = KFold(n_splits=5)
# algo = SVD(random_state=10)

# print('Sampled 10000 Ratings \n')

# number = 0
# for trainset, testset in kf.split(data):
#     algo.fit(trainset)
#     predictions = algo.test(testset)
#     precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)

#     # Precision and recall can then be averaged over all users
#     print('KFold Test #{}:'.format(number+1))
#     print('Precision: {}'.format(sum(prec for prec in precisions.values()) / len(precisions)))
#     print('Recall: {}'.format(sum(rec for rec in recalls.values()) / len(recalls)))
#     print('\n')
#     number += 1

# # Load full dataset
# data = Dataset.load_from_df(df[['customer_id', 'product_id', 'star_rating']], reader)
# kf = KFold(n_splits=5)
# algo = SVD(random_state=10)

# print('Full Dataset \n')
# number = 0
# for trainset, testset in kf.split(data):
#     algo.fit(trainset)
#     predictions = algo.test(testset)
#     precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)

#     # Precision and recall can then be averaged over all users
#     print('KFold Test #{}:'.format(number+1))
#     print('Precision: {}'.format(sum(prec for prec in precisions.values()) / len(precisions)))
#     print('Recall: {}'.format(sum(rec for rec in recalls.values()) / len(recalls)))
#     print('\n')
#     number += 1

# Create simple id to map products
df['id'] = df.groupby('product_id').ngroup()

# Train the algortihm to compute the similarities between items
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['customer_id', 'id', 'star_rating']], reader)
trainset = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)

# Read the mappings raw id <-> product name
rid_to_name = dict(zip(df.id, df.product_title))
name_to_rid = dict(zip(df.product_title, df.id))

# Read input from input.txt
with open('input.txt', 'r') as f:
    input = f.read()

# Retrieve inner id of the product
product_raw_id = name_to_rid[input]
product_inner_id = algo.trainset.to_inner_iid(product_raw_id)

# Retrieve inner ids of the nearest neighbors of Toy Story.
product_neighbors = algo.get_neighbors(product_inner_id, k=4)

# Convert inner ids of the neighbors into names.
product_neighbors = (algo.trainset.to_raw_iid(inner_id)
                       for inner_id in product_neighbors)
product_neighbors = (rid_to_name[rid]
                       for rid in product_neighbors)

with open('output.txt', 'w') as f:
    # Redirect stdout to the output file
    sys.stdout = f

    # print("The 4 most similar products to",input,"are:\n")
    ranking = 0
    for neighbor in product_neighbors:
        print(neighbor)
        ranking += 1


sys.stdout = original_stdout



# # Process the input_data and generate output_data
# # For example:
# output_data = input_data

# # Write the output_data to output.txt
# with open('output.txt', 'w') as f:
#     f.write(output_data)
