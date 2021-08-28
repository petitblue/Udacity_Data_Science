import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('cleaned_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message'] # message counts by genre
    genre_names = list(genre_counts.index)
    # message counts by category
    category_counts = df[df.columns[4:]].sum().sort_values(ascending=False)
    category_name = list(category_counts.index)
    # Genre distribution in Top 10 category
    category_lables = df[df.columns[4:]].sum().sort_values(ascending=False).index
    df_genre = df.groupby('genre')[category_lables].sum().reset_index()
    df_genre =df_genre.drop(columns=['genre']).rename(index={0 : 'direct', 1:'news', 2:'social'})
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'template': "seaborn"
            }
        },
        {
            'data': [
                Bar(
                    x=category_name,
                    y=category_counts

                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45
                },
                'template': "seaborn"
            }
        },
        {
            'data': [
               Bar(
                x=category_lables[:10],
                y=df_genre.iloc[0],
                name='Direct'
                ),
                Bar(
                    x=category_lables[:10],
                    y=df_genre.iloc[1],
                    name='News'
                ),
                Bar(
                    x=category_lables[:10],
                    y=df_genre.iloc[2],
                    name='Social'
                )
            ],
            'layout': {
                'title': 'Distribution of Messages by Genre in top 10 categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle': -45
                },
                'barmode': 'group'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()