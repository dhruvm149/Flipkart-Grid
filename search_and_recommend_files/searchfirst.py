import pandas as pd
import numpy as np

import warnings
import sys
original_stdout = sys.stdout

warnings.filterwarnings('ignore')

df_mean=pd.read_csv('df_mean.csv')


class Recommender:
    def __init__(self, n=5, adjusted_rating=True):
        
        self.n = n # Number of recommendations to return, default is 5
        self.adjusted_rating = adjusted_rating # Boolean determines if original star rating or adjust rating is used
        # Initiate product variables to display in recommendation results
        self.product_variables = ['product_id', 'product_title', 
                                'product_category', 'star_rating', 'adjusted_rating', 'purchased_counts']
        
        # Initiate list of recommendations to be sorted by rating scores, original or adjusted
        if self.adjusted_rating: # Set standard sorting criteria to adjusted rating
            rating = 'adjusted_rating'
        else: # Set sorting criteria to originial, or star rating
            rating = 'star_rating'
        self.recommend = df_mean.sort_values(rating, ascending=False)
        self.output=[]
        
        
    def _filter_by_product_category(self):

        
        idx = []
        for i in self.recommend.index: # Search through index
            if self.recommend.loc[i, 'product_category'] is not np.nan:
                keyword_search = self.recommend.loc[i, 'product_category'].split(',') # Locate index, product category
                if self.product_category.lower() in str(keyword_search).lower(): # Check if search item in keyword_search
                    idx.append(i) # Place index of row in a list
        self.recommend = self.recommend.loc[idx]
        
    def _filter_by_product_title(self):
        
        idx = []
        for i in self.recommend.index: # Search through index
            if self.recommend.loc[i, 'product_title'] is not np.nan:
                keyword_search = self.recommend.loc[i, 'product_title'].split(',') # Locate index, product category
                if self.product_title.lower() in str(keyword_search).lower(): # Check if search item in keyword_search
                    idx.append(i) # Place index of row in a list
        self.recommend = self.recommend.loc[idx] 
        
    def return_recommendations(self):
        
        if len(self.recommend) == 0:
            self.output.append('No products recommended.')
        elif self.n < len(self.recommend): # Returns top n products from list of recommendations
            # print('Top {} recommended products for you:'.format(self.n))
            top_n_titles = self.recommend['product_title'].iloc[:self.n].tolist()
            for title in top_n_titles:
                self.output.append(title)
            # print(self.recommend.iloc[:self.n][self.product_variables])
        else: # Returns all products if amount found is less than n
            # print('Top {} recommended products for you:'.format(len(self.recommend)))
            top_n_titles = self.recommend['product_title'].tolist()
            for title in top_n_titles:
                self.output.append(title)
            # print(self.recommend[self.product_variables])
    
    def return_keyword(self):
        
        if len(self.recommend) == 0:
            print('No products recommended.')
        elif 1 < len(self.recommend): # Returns top n products from list of recommendations
            # print('Top {} recommended products for you:'.format(self.n))
            top_n_titles = self.recommend['product_title'].iloc[:self.n].tolist()
            for title in top_n_titles:
                self.output.append(title)
        
            # print(self.recommend.iloc[:1][self.product_variables])
        else: # Returns all products if amount found is less than n
            # print('Top {} recommended products for you:'.format(len(self.recommend)))
            top_n_titles = self.recommend['product_title'].tolist()
            for title in top_n_titles:
                self.output.append(title)
            # print(self.recommend[self.product_variables])
            
    # Keyword search filtering recommender module
    def keyword(self, df=df_mean, product_title=None, product_category=None):
        self.output=[]
        
        self.recommend = df # Assign dataframe
        self.product_variables = ['product_id', 'product_title', 
                                  'product_category', 'star_rating', 'adjusted_rating', 'purchased_counts']
        
        # Assign variables based on user's keyword search
        self.product_title = product_title
        self.product_category = product_category
            
        # Filter by product title
        if self.product_title != None:
            self._filter_by_product_title()
            if len(self.recommend) == 0:
                # with open('output.txt', 'w') as f:
                #     sys.stdout = f
                self.output.append('No matching products found for {}'.format(self.product_title))
                    # print('No matching products found for {}'.format(self.product_title))
                # with open('output.txt', 'w') as f:
                #     sys.stdout = f
                #     print(self.output)
                return self.output
                
        # Filter by product category
        if self.product_category != None:
            self._filter_by_product_category()
            if len(self.recommend) == 0:
                self.output.append('No matching products found for {}'.format(self.product_category))
                # with open('output.txt', 'w') as f:
                #     sys.stdout = f
                    # print('No matching products found for {}'.format(self.product_category))
                return self.output
            
        # Sort by rating of interest
        if self.adjusted_rating:
            rating = 'adjusted_rating'
        else:
            rating = 'star_rating'
            
        self.recommend = self.recommend.sort_values(rating, ascending=False)
            
        # Return top n recommendations
        self.return_recommendations() 
        # with open('output.txt', 'w') as f:
        #     sys.stdout = f
        #     print(self.output)
        return self.output
    

def search1(input):
    # with open('input.txt', 'r') as f:
    #     input = f.read()
    kw = Recommender(n=4)
    
    return(kw.keyword(product_title=input))

# with open('output.txt', 'w') as f:
#         sys.stdout = f
#         print(search1())
# search1()
sys.stdout = original_stdout
