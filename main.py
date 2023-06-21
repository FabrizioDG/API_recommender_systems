import os
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
from flask import Flask, request, send_file, render_template, jsonify
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
import datetime
#import ast
#import sys

from data_preprocessing_utilities import *
from recommending_events_model import *
from global_variables import *


app = Flask(__name__)

'''
API creada para gestionar peticiones desde el Backend de la APP y sugerir eventos afines a los usuarios, así como sugerir usuarios afines entre sí.

Dispone de 3 endpoints:

- transfer_database: Conecta con la base de datos MongoDB (backend de la APP alojada en Atlas) y clona dichos registros en una base de datos Postgres (alojada en AWS),
con la finalidad de reeentrenar a los modelos de Machien Learning encargados de las sugerencias a usuarios.

- user_recommend: Recibe petición con los datos de usuario (relativos a su perfil, asi como sus skills y sus hobbies) y devuelve un listado de usuarios ordando por afinidad.

- events_recommend: Recibe petición con los datos de usuario (relativos a participación del usuario a eventos) y devuelve un listado de eventos afines según sus intereses
y bassándose en participación a eventos de usuarios afines.

'''

@app.route('/')
def index():
    return "<h1>La conexión está disponible</h1><p>Conexión en curso</p>"

@app.route('/events_recommendations')
def events_recommendations():
    '''
    API endpoint which connects to make predictions on the 3 most recommended events for each user of the
    Ágora app.

    Accepts GET requests with payload "update_AWS_DB" passed as parameter. The endpoint connects to mongoDB database and get tableswith users
    and events, transform the data in a suitable way for training together with artificial data, then train an 
    SVD matrix factorization model.
    Returns a JSON response with the predictions for all the users in the database. If update_AWS_DB=="yes",
    it save the results of the predictions with the date and time when the API has been called in a PostgreSQL
    database on AWS.
    '''

    update_AWS_DB = request.args.get('update_AWS_DB')
    
    (users_columns, users, events_columns, events,tags_columns, tags,_, _,
     _, _, _, _, _, _) = connection_db_mongodb()
    df_users = pd.DataFrame(users, columns=users_columns)
    df_events = pd.DataFrame(events, columns=events_columns)
    df_tags = pd.DataFrame(tags, columns=tags_columns)

    df_real_users, df_real_events = create_training_df_recommendation(df_users, df_events, df_tags)
    results = train_SVD_model(df_real_users, df_real_events)

    #I drop this column which is a timestamp. I don't need it and it causes problems
    df_events = df_events.drop(["updatedAt"], axis=1)
    df_events["attendees"] = df_events["attendees"].apply(lambda x : [str(z) for z in x]) 
    # Convert all the ObjectId objects in str to avoid problems when converting to json
    for i in range(len(df_events)):
        df_events.loc[i,"_id"] = str(df_events.loc[i,"_id"])
        tags_names = [df_tags[df_tags["_id"] == id_tag]["name"].values[0] for id_tag in df_events.loc[i,"eventTags"]]
        if len(tags_names)<=4:
            df_events.at[i,"eventTags"] = tags_names
        else:
            df_events.at[i,"eventTags"] = tags_names[0:4]
    #prepare the output for the API, FullStack wants a json with all the information of the first 3 recommended events for each user
    preds = {str(user) : {key : [] for key in df_events.columns} for user in df_real_users["id_user"].unique()}
    for user_id, event in results.items():
            for event_id in event.keys():
                row = df_events[df_events["_id"]==event_id].values[0]
                for i, key in enumerate(preds[user_id].keys()):
                    preds[user_id][key].append(row[i])
    # Organize the dictionary in a dataframe with columns id_user, id_event, score, to save it in the postgre DB
    rows = []
    for id_user, events in results.items():
        for id_event, score in events.items():
            rows.append({'id_user': id_user, 'id_event': id_event, 'score': score})
    df = pd.DataFrame(rows)
    dates = [datetime.datetime.now()]*len(df)
    df_dates = pd.DataFrame(dates, columns=["date"])
    df_final = pd.merge(df,df_dates, right_index=True, left_index=True)
    #If this parameter is not set to yes, do not save in the postgresql database on AWS
    if update_AWS_DB=="yes":
        engine_postgres = create_engine(os.getenv("URL_POSTGRESQL_AWS"))
        df_final.to_sql('events_recommendations', engine_postgres, if_exists='append', index=False)
        engine_postgres.dispose()
    return json.dumps(preds)

@app.route('/match_all_users')
def match_all_users():
    '''
    API endpoint which connects to make predictions on the 4 most similar matching users for each user of the
    Ágora app.

    Accepts GET requests with payload "update_AWS_DB" passed as parameter. The endpoint connects to mongoDB database and get tableswith users
    and events, transform the data in a suitable way for training together with artificial data, then calculate
    cosine_similarity to assess the similarity between the users in the platform. 
    Returns a JSON response with the predictions for all the users in the database. If update_AWS_DB=="yes",
    it save the results of the predictions with the date and time when the API has been called in a PostgreSQL
    database on AWS.
    '''
    update_AWS_DB = request.args.get('update_AWS_DB')
    (users_columns, users, _, _,_, _,degrees_columns, degrees,skills_columns,
      skills, hobbies_columns, hobbies, userTypes_columns, userTypes) = connection_db_mongodb()
    df_users = pd.DataFrame(users, columns=users_columns)
    df_degrees = pd.DataFrame(degrees, columns=degrees_columns)
    df_skills = pd.DataFrame(skills, columns=skills_columns)
    df_hobbies = pd.DataFrame(hobbies, columns=hobbies_columns)
    df_userTypes = pd.DataFrame(userTypes, columns=userTypes_columns)
    
    dataset = create_training_df_userMatching(df_users, df_hobbies, df_skills, df_degrees)
    
    # Definir una función que sustituya Femenino por O, Masculino por 1 y No especifica por 2 en la columna sexo
    def gender(x):
        if x == "Hombre":
            return 1
        elif x == "Mujer":
            return 0
        elif x == "No especifica":
            return 2

    dataset["sexo"] = dataset["sexo"].apply(lambda x: gender(x))
    #FDG: he tenido que añadir esto para quitar la cuestión del ObjectId
    dataset["id_user"] = dataset["id_user"].astype(str)
    users = dataset["id_user"].tolist()

    # Paso 1: Normalizar los valores en una escala de 0 a 1
    scaler = MinMaxScaler()

    # Crear el pipeline con el paso de normalización
    pipeline = Pipeline([
        ('scaler', scaler)
    ])

    users_afinidad = {}

    for user in users:

        #FDG: Lo he cambiado con esto porque me daba problemas cuando cargaba los datos desde mongo
        #FDG: NOTA tenemos que averiguar como va, porque también
        following = dataset[dataset["id_user"]==user]["following"].values[0]
        #Hay que añadir esto porque los elementos en la lista serán también ObjectId y darán problemas
        following = [str(x) for x in following]
        # Aplicar el pipeline y obtener los datos normalizados
        intereses_dataset_normalized = pipeline.fit_transform(dataset.drop(["id_user", "following"], axis=1))

        # Obtener el vector de intereses normalizado del usuario de referencia
        intereses_referencia = intereses_dataset_normalized[dataset["id_user"] == user]

        # Calcular la similitud de coseno entre el usuario de referencia y todos los demás usuarios
        similitudes = cosine_similarity(intereses_referencia, intereses_dataset_normalized)

        # Obtener los usuarios y sus respectivas similitudes
        usuarios = dataset["id_user"].values
        similitudes_usuarios = similitudes.flatten()

        # Crear un DataFrame con los usuarios y sus similitudes
        similitudes_df = pd.DataFrame({"Usuario": usuarios, "Similitud": similitudes_usuarios})

        # Filtrar los usuarios que ya están en el "following" del usuario actual
        similitudes_df = similitudes_df[~similitudes_df["Usuario"].isin(following)]

        # Obtener los usuarios más similares (máximo 5)
        usuarios_afines = similitudes_df.sort_values(by="Similitud", ascending=False).iloc[1:5]["Usuario"].tolist()

        users_afinidad[user] = usuarios_afines

    df = pd.DataFrame(users_afinidad.items(), columns=['username', 'users_affinity'])
    #Add the date to save in database
    dates = [datetime.datetime.now()]*len(df)
    df_dates = pd.DataFrame(dates, columns=["date"])
    df_final = pd.merge(df,df_dates, right_index=True, left_index=True)
 
    if update_AWS_DB=="yes":
        engine_postgres = create_engine(os.getenv("URL_POSTGRESQL"))
        df_final.to_sql('users_matching', engine_postgres, if_exists='append', index=False)
        engine_postgres.dispose()

    #REORGANIZE JSON FOR FULLSTACK
    df_users["_id"] = df_users["_id"].astype(str)
    preds = {str(user) : {key : [] for key in ["_id", "username", "degree", "userType"]} for user in df_users["_id"].unique()}
    for index, row in df.iterrows():
        user_id = row["username"]
        recommended_ids = row["users_affinity"]
        for rec_id in recommended_ids:
            username = df_users[df_users["_id"] ==rec_id]["username"].values[0]
            degree_id = df_users[df_users["_id"] ==rec_id]["degree"].values[0]
            degree = df_degrees[df_degrees["_id"]==degree_id]["name"].values[0]
            userType_id = df_users[df_users["_id"] ==rec_id]["userType"].values[0]
            userType = df_userTypes[df_userTypes["_id"]==userType_id]["name"].values[0]
            preds[user_id]["_id"].append(rec_id)
            preds[user_id]["username"].append(username)
            preds[user_id]["degree"].append(degree)
            preds[user_id]["userType"].append(userType)

    results_json = json.dumps(users_afinidad, ensure_ascii=False, indent=4)
    json_object = json.loads(results_json)

    # Devolver la respuesta en formato JSON
    return preds #jsonify(json_object)

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))