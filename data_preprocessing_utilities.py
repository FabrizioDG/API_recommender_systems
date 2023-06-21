import numpy as np
import random
import pymongo
import pandas as pd
from dotenv import load_dotenv
import os
from global_variables import * 

def configure():
  '''
  Load dotenv to load the mongodb url from a env file (for security reasons)
  '''
  load_dotenv()

def gr(result):
    '''
    Function to get all the data from a mongodb collection in a shape that can be transformed into a pandas DataFrame
    Input:
    result: pymongo.collection.Collection
      The collection we want to convert to pandas df
    '''
    return [x for x in result]

def connection_db_mongodb():
    '''
    Connect to the mongodb database at URL_MONGODB and extract the relevant information from the collections
    users, events, tags, degrees, skills, hobbies. It returns the collections and the column names as tuples.

    OSS: An error occurs only when this function is called on the Railway and AWS Server at the moment of returning
    pandas dataframe as outputs. We do not understand why this happens, it must be some encoding problem. To
    solve this problem we return the dataframe in the format of a list, and then we transform it back to DataFrame
    outside of the function. (Yes, it seems crazy we are also confused!)

    Output:
    df_users.columns: tuple
      list of the columns of the collection df_users
    df_users.values: tuple
      list of the values of the collection df_users
    df_events.columns: tuple
      list of the columns of the collection df_events
    df_events.values: tuple
      list of the values of the collection df_events
    df_tags.columns: tuple
      list of the columns of the collection df_tags
    df_tags.values: tuple
      list of the values of the collection df_tags
    df_degrees.columns: tuple
      list of the columns of the collection df_degrees
    df_degrees.values: tuple
      list of the values of the collection df_degrees
    df_skills.columns: tuple
      list of the columns of the collection df_skills
    df_skills.values: tuple
      list of the values of the collection df_skills
    df_hobbies.columns: tuple
      list of the columns of the collection df_hobbies
    df_hobbies.values: tuple
      list of the values of the collection df_hobbies
    df_usertypes.columns: tuple
      list of the columns of the collection df_usertypes
    df_usertypes.values: tuple
      list of the values of the collection df_usertypes
    '''
    #Connection to mongoDB Database
    configure()
    client = pymongo.MongoClient(os.getenv('URL_MONGODB'))
    db = client.app_dt

    collection = db.users
    projection = {'_id': 1, 'suscriptions': 1, 'gender' : 1, 'degree' : 1, 'age' : 1, 'following' : 1,
                   'skills' : 1, 'hobbies' : 1, "userType" : 1, "username" : 1}
    df_users = pd.DataFrame(gr(collection.find({}, projection)))

    collection = db.usertypes
    projection = {'_id': 1, 'name' : 1}
    df_usertypes = pd.DataFrame(gr(collection.find()))

    collection = db.events
    df_events = pd.DataFrame(gr(collection.find()))

    collection = db.tags
    projection = {'_id': 1, 'name': 1}
    df_tags = pd.DataFrame(gr(collection.find({}, projection)))

    collection = db.degrees
    projection = {'_id': 1, 'name': 1}
    df_degrees = pd.DataFrame(gr(collection.find({}, projection)))

    collection = db.skills
    projection = {'_id': 1, 'name': 1}
    df_skills = pd.DataFrame(gr(collection.find({}, projection)))

    collection = db.hobbies
    projection = {'_id': 1, 'name': 1}
    df_hobbies = pd.DataFrame(gr(collection.find({}, projection)))

    # I drop nans, because some users,events might not have all the information I need
    # For instance some of the users created by FS developers for testing
    df_users = df_users.dropna().reset_index(drop=True)
    df_events = df_events.dropna().reset_index(drop=True)
    return (tuple(df_users.columns), tuple(df_users.values),
           tuple(df_events.columns), tuple(df_events.values),
           tuple(df_tags.columns), tuple(df_tags.values),
           tuple(df_degrees.columns), tuple(df_degrees.values),
           tuple(df_skills.columns), tuple(df_skills.values),
           tuple(df_hobbies.columns), tuple(df_hobbies.values),
           tuple(df_usertypes.columns), tuple(df_usertypes.values)
    )


def create_training_df_recommendation(df_users, df_events, df_tags):
  '''
  Reorganize the information in the collections df_users, df_events, df_tags extracted from mongoDB in two pandas
  Dataframes that have the same format which is used for training the Recommendation model for recommending events to
  the users.

  Inputs:
  df_users: pandas.Dataframe
     The pandas dataframe from the collection users 
  df_events: pandas.Dataframe
     The pandas dataframe from the collection events
  df_tags: pandas.Dataframe
     The pandas dataframe from the collection tags 

  Output:

  df_users_new: pandas.DataFrame
    A pandas dataframe for the users and their participation to events,
    in the format used for training the recommending system
  df_events_new: pandas.DataFrame
    A pandas dataframe for the events and their tags,
    in the format used for training the recommending system
  '''

  dictio_users = {"id_user" : [], "id_event" : [], "participation" : [], "Desarrollo profesional" : [], "Negocios y Finanzas" : [],
                  "Innovación y Tecnologia" : [],	"Sostenibilidad" : [],	"Recursos Humanos": [],	"Estilo de vida": [],
                    "Arte y Cultura": [],	"Aprendizaje y educación": [],	"Gaming": []}
  df_users_new = pd.DataFrame(dictio_users)
  #tmp dataframes to organize the information related to the users
  df_tmp_participation = pd.DataFrame({"id_user" : [], "id_event" : [], "participation" : []})
  df_tmp_tags = pd.DataFrame({elem : [] for elem in cluster_tags})
  #df for the events
  df_events_new = pd.DataFrame(dict({"id_event" : [], "time" : []}, **{elem : [] for elem in cluster_tags}))
  #Loop over the df_users and df_events
  for index_user, row_user in df_users.iterrows():
    #cluster_tags son los tags definidos al principio
    dictio_cluster_tags = {elem : 0 for elem in cluster_tags}
    for index_event, row_event in df_events.iterrows():
      #this is the index of each new row of the output dataframe
      index = int(index_user*len(df_events)) + index_event
      id_user = row_user["_id"]
      id_event = row_event["_id"]
      time_event = row_event["time"]
      #Assign all 0 to the tag events
      df_events_new.loc[index_event, "id_event"] = id_event
      df_events_new.loc[index_event, "time"] = time_event

      df_events_new.iloc[index_event, 2:] = 0
      #now let's store the tags of the event at which the user put his/her participation
      #first I collect the tags_ids of the event where the user participated
      tags_ids = df_events[df_events["_id"]==id_event]["eventTags"].values[0]
      for tag_id in tags_ids:
          #then for each of this tag_id I recover the name of that tag and I add
          tag_name = df_tags[df_tags["_id"]==tag_id]["name"].values
          
          for key, tags in mapping_tags.items():
            if tag_name in [x for x in tags]:
                df_events_new.loc[index_event, key] += 1
          #if the id_event is in the subscriptions column for user id_user, then participation = 1,
          #and map the tag_name into the cluster_tags names and add +1 to the corresponding one
          if id_event in row_user["suscriptions"]:
            participation = 1
            #Now map the tag_name into the cluster_tags names and add +1 to the corresponding one
            for key, tags in mapping_tags.items():
              if tag_name in [x for x in tags]:
                  dictio_cluster_tags[key] += 1
          else:
            participation = 0
      

      df_tmp_participation.loc[index] = [id_user, id_event, participation]
    #Not sure why but it saves the columns as floats and not ints, so I do this
    df_events_new[list(df_events_new.columns[2:])] = df_events_new[list(df_events_new.columns[2:])].astype(int)
    
    for i in range(len(df_events)):
      df_tmp_tags.loc[int(index_user*len(df_events))+i] = dictio_cluster_tags.values()
    df_users_new = pd.merge(left=df_tmp_participation, right=df_tmp_tags, left_index=True, right_index=True)
  return df_users_new, df_events_new



def create_training_df_userMatching(df_users, df_hobbies, df_skills, df_degrees):
  '''
  Reorganize the information in the collections df_users, df_hobbies, df_skills, df_degrees extracted from mongoDB 
  in a dataframe that have the same format which is used for training the Users Matching algorithm.

  Inputs:
  df_users: pandas.Dataframe
     The pandas dataframe from the collection users 
  df_hobbies: pandas.Dataframe
     The pandas dataframe from the collection hobbies
  df_skills: pandas.Dataframe
     The pandas dataframe from the collection skills 
  df_degrees: pandas.Dataframe
     The pandas dataframe from the collection degrees 
  Output:

  df_users_matching: pandas.DataFrame
    A pandas dataframe for the users with age, gender, degree (1-hot encoding), skills (1-hot encoding),
    hobbies (1-hot encoding), which is the format used for training the user matching algorithm.
  '''
  dictio_users = dict({"id_user" : [], "edad" : [], "sexo" : [], "following" : []},
                       **{elem : [] for elem in df_degrees["name"].unique()},
                    **{elem : [] for elem in df_hobbies["name"].unique()}, **{elem : [] for elem in df_skills["name"].unique()})
  df_users_matching = pd.DataFrame(dictio_users)
  
  #Loop over the df_users and df_events
  for index_user, row_user in df_users.iterrows():
    id_user = row_user["_id"]
    age = row_user["age"]
    gender = row_user["gender"]

    degree_id = row_user["degree"]
    skills_ids = row_user["skills"]
    hobbies_ids =row_user["hobbies"]
    following = row_user["following"]
    degree_name = df_degrees[df_degrees["_id"]==degree_id]["name"].values[0]
    hobbies_names = []
    skills_names = []

    for hobbie_id in hobbies_ids:
      hobbie_name = df_hobbies[df_hobbies["_id"]==hobbie_id]["name"].values[0]
      hobbies_names.append(hobbie_name)
    for skill_id in skills_ids:
      skill_name = df_skills[df_skills["_id"]==skill_id]["name"].values[0]
      skills_names.append(skill_name)

    # one-hot encodings
    onehot_degrees = []
    onehot_skills = []
    onehot_hobbies = []

    for degree in df_degrees["name"].unique():
      if degree==degree_name:
        onehot_degrees.append(1)
      else:
        onehot_degrees.append(0)

    for skill in df_skills["name"].unique():

      if skill in skills_names:
        onehot_skills.append(1)
      else:
        onehot_skills.append(0)


    for hobbie in df_hobbies["name"].unique():
      if hobbie in hobbies_names:
        onehot_hobbies.append(1)
      else:
        onehot_hobbies.append(0)
    total_list = [id_user] + [age] + [gender] + [following] + onehot_degrees + onehot_hobbies + onehot_skills
    df_users_matching.loc[index_user] = np.asarray(total_list, dtype=object)
  
  return df_users_matching


def create_artificial_users():
  '''
  This function create artificial users and events to train the recommendation models. We build 50 fake events,
  and 1000 fake users.

  Output:
   A dataframe in the form of user-item matrix, as needed by the SVD matrix factorization model.
  
  '''
  random.seed(42)

  # Generar el DataFrame
  tabla = []

  np.random.seed(42)

  for i in range(50):
      #evento = {"Evento": "Evento " + str(i + 1)}
      evento = {"id_event": i+1}

      etiquetas_asignadas = random.sample(cluster_tags, 3)
      for elemento in cluster_tags:
          evento[elemento] = 1 if elemento in etiquetas_asignadas else 0
      tabla.append(evento)

  df_eventos = pd.DataFrame(tabla)

  np.random.seed(42)  # Para reproducibilidad de los datos aleatorios

  # Crear diccionario de etiquetas para asignar a los eventos
  diccionario_eventos = {}
  #Las categorias son las mismas columnas de los tags en el df_eventos
  categorias = list(df_eventos.columns[1:])

  # Generar datos para el dataframe
  data = []
  for i in range(1000):
      id_usuario = i + 1
      #generar el número de eventos al cual el usuario participa, sacados de una distribución normal de media 15 y sigma 5
      num_eventos = int(np.random.normal(15, 5))  
      #OSS si el numero de eventos es 0 da problemas, pues pongo que minimo ha asistido a 1
      if num_eventos==0:
          num_eventos=1
      # Generar a cual eventos aleatorios el usuario ha participado, dependiendo de num_eventos
      deck = list(range(0,len(df_eventos)))
      np.random.shuffle(deck)
      #index eventos es un array que tendrá los id_events (en realidad id_events-1) a los cuales el usuario ha participado
      index_eventos =np.array(deck[0:num_eventos])
      #pongo 1 solo a los eventos al cual el usuario ha participado
      participacion_eventos = np.zeros(len(df_eventos),dtype=int)
      participacion_eventos[index_eventos]=1
      #Ahora quiero obtener los tags de cada evento al cual el usuario ha participado
      #Antes de todo defino los id de los eventos al cual el usuario ha participado, que serán index_eventos + 1
      #Eso porque index_eventos son valores entre 0 y 49, y los id_eventos los hemos puestos empezando de 1 y no 0
      id_eventos = index_eventos + 1

      #Ahora voy al df_eventos y le pongo mascara para selecionar solo los eventos con los id_event a los cuales el usuario
      # ha participado. Seleciono solo las columnas de los tags (iloc[:,1:]), y sumo todos los valores que hay en cada columna.
      # Me quedo con un DataSeries con los tags y los valores de las sumas, yo solo quiero los valores y le pongo values.
      valores_categorias = df_eventos[df_eventos["id_event"].isin(id_eventos)].iloc[:,1:].sum().values
      eventos_categoria = {categorias[i] : valores_categorias[i] for i in range(len(categorias))}
  
      if num_eventos < 0:
          num_eventos = 0
      estudio = np.random.choice(["Grado", "Master", "Bootcamp"], p=[0.73, 0.21, 0.06])
      if estudio == "Grado":
          edad = np.random.randint(18, 26)
          sexo = np.random.choice(["Hombre", "Mujer"], p=[0.64, 0.36])
      elif estudio == "Master":
          edad = np.random.randint(24, 31)
          sexo = np.random.choice(["Hombre", "Mujer"], p=[0.66, 0.34])
      else:  # Bootcamp
          edad = np.random.randint(19, 46)
          sexo = np.random.choice(["Hombre", "Mujer"], p=[0.81, 0.19])
      emprende = np.random.choice([0, 1], p=[0.4, 0.6])

      for j in range(len(df_eventos)):
          row = [id_usuario, j+1, num_eventos, estudio, edad, sexo, emprende]
          for categoria in categorias:
              row.append(eventos_categoria[categoria])
          row.append(participacion_eventos[j])
          data.append(row)

  # Crear dataframe
  columnas = ["id_user", "id_event", "num_eventos", "estudio", "edad", "sexo", "emprende"]
  columnas += categorias
  columnas += ["participation"]
  df = pd.DataFrame(data, columns=columnas)
  return df
