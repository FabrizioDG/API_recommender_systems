# Recommender ML models
This is a Python Flask code for the implementation of event and users recommender systems for a social network app. This is part of the team project we developed in the last 2 weeks of the Data Science bootcamp, in collaboration with students from the UX/UI, Full-stack development and cybersecurity bootcamps.

The url of the API is: http://13.38.31.251/ (NOTE: not deployed anymore, so the link is not active at the moment)

In the home there is nothing, just go directly to the endpoints

## Events recommendations

#### Model description

For recommending events to users in our app, we considered matrix factorization algorithms, and we used the library surprise scikit ([documentation](https://surpriselib.com/)) for their implementation. In particular we considered the Single Value Decomposition (SVD) model. These models take as an input a user-item matrix where the interaction between users and the product to recommend are stored (the product are events in the platform in our case); the element of the matrix should be ratings, and the supervised component of the algorithm is optimized on regression metrics (RMSE, MAE). In the case where explicit ratings are available, the model is from one side a supervised technique because its optimization is based on recovering the original user-item matrix, and from another side it is an unsupervised technique because it learns implicit patterns in the data, recognizing similarity between users and between items and use this information for the recommendations.

As for our problem we do not have rating but we have a binary classification problem (user participate or not to events), we reformulate the problem to adapt it to the models we used. The events in our platform are categorized by tags, so we built the user-item matrix counting the participation of the users to events with the same tag. For example, user1 participate to event1, event5, event6 which have certain tags; counting those tags we obtain that user1 participated to 3 "Innovation and Technology", 2 "Gaming", 2 "Business and finance", and 1 "Professional Development" events. The row in the matrix corresponding to user1 will have values 3,2,2,1 to the corresponding tags, and 0 to the other possible tags.

Once the user-event matrix is built, we train the model, we store the result of the reconstructed user-event matrix, and this reconstructed matrix will tell us the preferences of each user related to the different event tags. As we want to recommend specific events, we take this result and map it into a result onto events, simply averaging the assigned score to each tag of each event. For example, same user1 as before got in the reconstructed matrix 2.7, 1.8, 1.6, 1.5 for the tags "Innovation and Technology", "Gaming", "Professional Development", and "Business and finance" respectively. If for example event3 has the tags "Innovation and Technology" and " Business and Finance" its final score is going to be $\frac{2.7 + 1.5}{2} = 2.1$; we then recommend the events at which the user didn't participate (or intend to participate) which have the higher score.

#### Important INTERPRETATION

We report here as an example the result of the recommendation of our model on one artificial user we created for the training. More details in the notebook. In the original matrix, the top categories of event for user "6483c9bc6d386542ccf97cb0" are:

- Innovación y Tecnologia: 7
- Negocios y Finanzas: 3
- Sostenibilidad: 3
- Estilo de vida:	3
- Desarrollo profesional:	1

The predictions of our model instead gives us the scores:
- 'Innovación y Tecnologia', 4.345545503223931
- 'Estilo de vida', 2.567346030959505
- 'Desarrollo profesional', 2.4003032347528066
- 'Negocios y Finanzas', 2.3408710278004747

So for instance the model recognized that this user might be interested in "Desarrollo profesional", even though in the original matrix his/her participation was lower than other type of events.

This is due to collaborative filtering, the algorithms recognize patterns between different users and suggest this particupar user possible interests, due to his/her similarity with other users.

#### Endpoint events_recommendations

url : http://13.38.31.251/events_recommendations

This is a GET request, and you don't need to pass anything. There is only a parameter which if set to yes it will save the results in a database in AWS. There is no need to use it for now.

This endpoint will return a json which contains for each user his/her best 3 recommended events, with all the information about title, description, tags, date, place, ready to be depicted in the app.
  
As an example of response of the API:
```
{
    "648344339e7a3b1512b5998f": {
        "_id": [
            "648349af53f64d7139943754",
            "648349af53f64d7139943751",
            "648349af53f64d7139943758"
        ],
        "time": [
            "06-01-2024",
            "02-15-2024",
            "10-10-2024"
        ],
        "eventTags": [
            [
                "Artes visuales",
                "Innovación",
                "Sostenibilidad",
                "Música",
                "Cultura"
            ],
            [
                "Tecnología",
                "Innovación",
                "Sostenibilidad",
                "Inteligencia Artificial",
                "Realidad Virtual"
            ],
            [
                "Inteligencia Artificial",
                "Realidad Virtual",
                "Tecnología",
                "Innovación",
                "Sostenibilidad"
            ]
        ],
        "event_name": [
            "Exposición de Artes Visuales",
            "Workshop de Tecnología",
            "Congreso de Inteligencia Artificial y Realidad Virtual"
        ],
        "description": [
            "Explora las últimas tendencias en arte visual y admira las obras de talentosos artistas contemporáneos.",
            "Explora las últimas tendencias tecnológicas y aprende a aplicarlas en tu empresa o proyecto personal.",
            "Explora los avances más recientes en inteligencia artificial y realidad virtual y descubre sus aplicaciones."
        ],
        "place": [
            "Galería de Arte XYZ",
            "Centro de Innovación XYZ",
            "Centro de Innovación XYZ"
        ]
    },
    "648379cc4099083d461dff3d": {
        "_id": [
            "648349af53f64d7139943758",
            "648349af53f64d713994375a",
            "648349af53f64d7139943754"
        ],
        "time": [
            "10-10-2024",
            "11-15-2024",
            "06-01-2024"
        ],
        "eventTags": [
            [
                "Inteligencia Artificial",
                "Realidad Virtual",
                "Tecnología",
                "Innovación",
                "Sostenibilidad"
            ],
            [
                "Ciberseguridad",
                "Blockchain",
                "Tecnología",
                "Innovación",
                "Sostenibilidad"
            ],
            [
                "Artes visuales",
                "Innovación",
                "Sostenibilidad",
                "Música",
                "Cultura"
            ]
        ],
        "event_name": [
            "Congreso de Inteligencia Artificial y Realidad Virtual",
            "Conferencia de Ciberseguridad y Blockchain",
            "Exposición de Artes Visuales"
        ],
        "description": [
            "Explora los avances más recientes en inteligencia artificial y realidad virtual y descubre sus aplicaciones.",
            "Aprende sobre las últimas técnicas de ciberseguridad y descubre el potencial de la tecnología blockchain.",
            "Explora las últimas tendencias en arte visual y admira las obras de talentosos artistas contemporáneos."
        ],
        "place": [
            "Centro de Innovación XYZ",
            "Auditorio XYZ",
            "Galería de Arte XYZ"
        ]
    }}

```
## Users recommendations

#### Model description

This model for matching users in our app is simply based on the cosine similarity between users. At the moment of creating their account, the users set some information on their profile regarding their demographic data and information about their skills/interests. The algorithm does an encoding of each user into vectors in the features space, and evaluate the cosine similarity between these vectors, to suggest to each user the other users which have the most similar profile to them.

#### Endpoint match_all_users

url : http://13.38.31.251/match_all_users

This is a GET request, and you don't need to pass anything. There is only a parameter which if set to yes it will save the results in a database in AWS. There is no need to use it for now.

This endpoint will return a json which contains for each user his/her best 4 recommended users, with all the information about name, degree, and type of user, ready to be depicted in the app.
  
As an example of response of the API:
```
{
    "648344339e7a3b1512b5998f": {
        "_id": [
            "648384abdfa2a8f0f9c6f717",
            "648379cc4099083d461dff3d",
            "64839594561db31d89f19635",
            "6483a6236d386542ccf97c95"
        ],
        "degree": [
            "Grado IGE, Ingeniería y Gestión Empresarial",
            "Grado IGE, Ingeniería y Gestión Empresarial",
            "Grado IGE, Ingeniería y Gestión Empresarial",
            "Grado IGE, Ingeniería y Gestión Empresarial"
        ],
        "userType": [
            "Usuario EDEM",
            "Usuario EDEM",
            "Usuario EDEM",
            "Usuario EDEM"
        ],
        "username": [
            "Pedro Rodríguez",
            "Juan Pérez",
            "David García",
            "Alejandro Torres"
        ]
    },
    "648379cc4099083d461dff3d": {
        "_id": [
            "6483a6236d386542ccf97c95",
            "6483c9bc6d386542ccf97cb0",
            "64839594561db31d89f19635",
            "648344339e7a3b1512b5998f"
        ],
        "degree": [
            "Grado IGE, Ingeniería y Gestión Empresarial",
            "Bootcamp Full Stack",
            "Grado IGE, Ingeniería y Gestión Empresarial",
            "Grado IGE, Ingeniería y Gestión Empresarial"
        ],
        "userType": [
            "Usuario EDEM",
            "Usuario EDEM",
            "Usuario EDEM",
            "Usuario EDEM"
        ],
        "username": [
            "Alejandro Torres",
            "Alejandro Gómez",
            "David García",
            "Guille"
        ]
    }}

```
