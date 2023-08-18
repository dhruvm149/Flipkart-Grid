import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# original_stdout = sys.stdout
from collections import defaultdict, Counter

from surprise import Reader, Dataset, KNNBaseline
# from surprise import SVD, SVDpp, NMF, SlopeOne, NormalPredictor, accuracy, KNNBasic
# from surprise import KNNWithMeans, BaselineOnly, CoClustering

# from surprise.model_selection import GridSearchCV, KFold
# from surprise.model_selection import train_test_split
# from surprise.model_selection import cross_validate

def search(input):
    # Your main program logic goes here
    # original_stdout = sys.stdout

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

    ranking = 0
    finalOutput = []
    for neighbor in product_neighbors:
        finalOutput.append(neighbor)
        # print(neighbor)
        ranking += 1

    # with open ('output.txt', 'w') as f:
    #     sys.stdout = f
    print(finalOutput)
    return finalOutput

# sys.stdout = original_stdout
# if __name__ == "__main__":
#     app.run()