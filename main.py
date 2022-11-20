

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import ipywidgets as widgets
from IPython.display import display


stream_set = pd.read_csv("final_data.csv")

ratings = pd.read_csv("ratings.csv")



def ez2read_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title


stream_set["clean_title"] = stream_set["title"].apply(ez2read_title)




vectorial_valuation = TfidfVectorizer(ngram_range=(1,2))

matrix_result = vectorial_valuation.fit_transform(stream_set["clean_title"])


def search_engine(title):
    title = ez2read_title(title)

    query_vectorial = vectorial_valuation.transform([title])
    comparison = cosine_similarity(query_vectorial, matrix_result).flatten()

    indices = np.argpartition(comparison, -5)[-5:]
    results = stream_set.iloc[indices].iloc[::-1]

    return results




stream_input = widgets.Text(
    value='Star Wars',
    description='Movie Title:',
    disabled=False
)
movie_selection = widgets.Output()

def type_func(data):
    with movie_selection:
        movie_selection.clear_output()
        title = data["new"]
        if len(title) > 3.5:
            display(search_engine(title))

stream_input.observe(type_func, names='value')

display(stream_input, movie_selection)





def recommendation_engine(movie_id):
    user_bubble = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 3)]["userId"]
    bubble_recommendation = ratings[(ratings["userId"].isin(user_bubble)) & (ratings["rating"] > 3)]["movieId"]
    bubble_recommendation = bubble_recommendation.value_counts() / len(user_bubble)

    bubble_recommendation = bubble_recommendation[bubble_recommendation > .10]
    full_user_pool = ratings[(ratings["movieId"].isin(bubble_recommendation.index)) & (ratings["rating"] > 3)]
    full_pool_general_recommendations = full_user_pool["movieId"].value_counts() / len(full_user_pool["userId"])
    relative_recommendations = pd.concat([bubble_recommendation, full_pool_general_recommendations], axis=1)
    relative_recommendations.columns = ["similarity", "all"]

    relative_recommendations["total"] = relative_recommendations["similarity"] / relative_recommendations["all"]
    relative_recommendations = relative_recommendations.sort_values("total", ascending=False)

    x = relative_recommendations.head(10).merge(stream_set, left_index=True, right_on="movieId")

    return x




stream_name_input2 = widgets.Text(
    value='Ted',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

def type_func(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 0:
            results = search_engine(title)
            movie_id = results.iloc[0]["movieId"]
            display(recommendation_engine(movie_id))

stream_name_input2.observe(type_func, names='value')






