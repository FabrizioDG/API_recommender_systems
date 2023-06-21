import random
import pandas as pd
import numpy as np
import datetime
import json
import pickle

from collections import defaultdict #data colector
#Surprise: https://surprise.readthedocs.io/en/stable/
import surprise
from surprise.reader import Reader
from surprise import Dataset
from surprise.model_selection import GridSearchCV
##CrossValidation
from surprise.model_selection import cross_validate
##Matrix Factorization Algorithms
from surprise import SVD
#from surprise import NMF
#from surprise import KNNBasic, KNNWithZScore, KNNWithMeans

from global_variables import *
from data_preprocessing_utilities import *

def train_SVD_model(df_real_users, df_real_events):
    '''
    Gridsearch for the SVD model for recommending events to users and makes prediction with the best model. 
    It returns a dictionary with the results of recommendations for all the users in the mongoDB database.

    Input:
    - df_real_users : pandas.DataFrame
        Dataframe with the user-item matrix obtained from the mongoDB database of the app.
    - df_real_events : pandas.DataFrame
        Dataframe with the events collection from the mongoDB database of the app.

    Output:
    - results : dict
      A dictionary with the results from the recommendation model with the best parameter from GridSearch.
    '''
    np.random.seed(42) # replicating results
    
    #load artificial data
    #df_users_artificial = pd.read_csv("df_artificial_users.csv", encoding="utf-8")
    df_users_artificial = create_artificial_users()
    #df_events_artificial = pd.read_csv("df_artificial_events.csv")
    
    #joint artificial and real data
    #df_events_total = pd.concat((df_events_artificial,df_real_events.drop(["time"], axis=1))).reset_index(drop=True)
    df_users_total = pd.concat((df_users_artificial.drop(["num_eventos", "estudio", "edad", "sexo", "emprende"],axis=1),
                                df_real_users)).reset_index(drop=True)
    df_users = df_users_total.set_index("id_user")
    #Only select the columns with the tags
    V = df_users.iloc[:,1:-1].drop_duplicates()
    user_event_matrix = V.reset_index()
    # Reshape the DataFrame to long format
    df = pd.melt(user_event_matrix, id_vars='id_user', var_name='event', value_name='participation')
    df.columns = ['user', 'event', 'participation']

    #Get the max value of participation of the entire dataframe of users to set the rating_scale in the Reader function
    max_value = df["participation"].max()
    
    reader = Reader(rating_scale=(0, max_value))
    data = Dataset.load_from_df(df=df, reader=reader)

    ##GridSearch using SVD model

    param_grid = {'n_factors': range(0,50)}
    gs_svd = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
    gs_svd.fit(data)
    # Save historic of best score and best parameters
    summary = {
        f"{datetime.datetime.now().date()}":{
            "model" : f"SVD model GridSearch",
            "best_rmse" : gs_svd.best_score['rmse'],
            "best_mae" : gs_svd.best_score['mae'],
            "best_params" : gs_svd.best_params['rmse']
        }
    }

    json_summary = json.dumps(summary)
    with open(f"history_models.json", "a") as file:
        file.write(json_summary)
    # Save most recent model in pickle
    with open('models/model_SVD_recommendations.pkl', 'wb') as file:
        pickle.dump(gs_svd, file)

    #PREDICTIONS only on the real users and real events
    
    results = make_predictions(df_real_users, df_real_events, data, 19)
                                #gs_svd.best_params['rmse']["n_factors"])
    return results

def make_predictions(df_real_users, df_real_events, data, n_factors):
    '''
    Function to make predictions using SVD model.

    Input: 
    - df_real_users : pandas.DataFrame
        Dataframe with the user-item matrix obtained from the mongoDB database of the app.
    - df_real_events : pandas.DataFrame
        Dataframe with the events collection from the mongoDB database of the app.
    - data : surprise.Dataset
      This is the Dataset object used in the surprise library to train the models.
    - n_factors: int
      Parameter of the SVD model, obtained after GridSearch

    Output:
    - scores : dict
      A dictionary with the predictions from the recommendation model with k = n_factors.
    '''
    #print("df_real_users", df_real_users[df_real_users["id_user"]=="648379cc4099083d461dff3d"])
    trainset = data.build_full_trainset()
    #This is how to build a testset for the predictions on the same dataset
    testset = data.build_full_trainset()
    testset = testset.build_testset()
    model = SVD(n_factors)
    model.fit(trainset)
    predictions = model.test(testset)
    #get the top n tags scores in descending order (n=9 by default as the number of tags used for training)
    top_n_clusterTags = get_top_n(predictions)
    # To recommend events to each user, we consider only future events, and events to which the user
    # didn't put his/her participation
    df_future_events = df_real_events[pd.to_datetime(df_real_events["time"])>datetime.datetime.now()]
    df_users_futureEvents = df_real_users[df_real_users["id_event"].isin(df_future_events["id_event"].values)]
    df_users_predictions = df_users_futureEvents[df_users_futureEvents["participation"]==0]
    #We want to build a dictionary where the keys are user_id  and the values are dictionaries with keys the events_id and value
    #the total score assigned to each event
    scores = {str(user) : [] for user in df_real_users["id_user"].unique()}

    #for each user and event in the final df
    for _, row_user in df_users_predictions.iterrows():
        user = row_user["id_user"]
        event = row_user["id_event"]
        #take only the event in the df_events_mongoDB dataframe
        df_ev_filtered = df_real_events[df_real_events["id_event"]==event]
        #for each original tag get the corresponding cluster_tag
        tot_tags = 0
        tot_score = 0
        for tag in df_ev_filtered.columns[2:]:
            #How many tags from this cluster_tags this event has?
            tag_weight = df_ev_filtered[tag].values[0]
            tot_tags += tag_weight
            print("user", user)
            print("tag", tag)
            
            #print(top_n_clusterTags[user])
            #print( top_n_clusterTags[user][tag])
            tag_score = top_n_clusterTags[user][tag]
            tot_score += tag_weight*tag_score
        if tot_tags!=0:
            scores[str(user)].append((str(event), tot_score/tot_tags))
    #sort the values
    for user, score in scores.items():
        score.sort(key = lambda x: x[1], reverse = True)
        #top_n[uid] = user_ratings[:n]
        scores[user] = dict(score[:3])
    return scores


def get_top_n(predictions, n = 9):
    '''Return the top n (default = 9) events tags for a user
    Input:
    - predictions: list
      List with the predictions from SVD model.
    - n : int
      how many event tags in the output list
    Output:
    - top_n : dict
      Sorted predictions, up to n elements. 
  
    '''
    #Part I.: Surprise docomuntation
    
    #1. First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    #2. Then sort the predictions for each user and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        #top_n[uid] = user_ratings[:n]
        top_n[uid] = dict(user_ratings[:n])
    
    return top_n