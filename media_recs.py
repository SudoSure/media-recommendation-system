import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import customtkinter as ctk
import tkinter as tk
import gzip

def load_imdb_data(filename):
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        return pd.read_csv(f, delimiter='\t', dtype={'startYear': 'str', 'endYear': 'str'})

# Load datasets
df_basics = load_imdb_data('title.basics.tsv.gz')
df_ratings = load_imdb_data('title.ratings.tsv.gz')
df_akas = load_imdb_data('title.akas.tsv.gz')

# Merge basics with ratings on tconst
df_movies = pd.merge(df_basics, df_ratings, on='tconst')

# Optionally merge with akas for alternative titles
df_movies = pd.merge(df_movies, df_akas[['titleId', 'title']], left_on='tconst', right_on='titleId', how='left')

# Filter for movies (excluding TV shows, shorts, etc.)
df_movies = df_movies[df_movies['titleType'] == 'movie']

# Filter for non-adult movies
df_movies = df_movies[df_movies['isAdult'] == 0]

# Drop unnecessary columns
df_movies = df_movies[['tconst', 'primaryTitle', 'originalTitle', 'startYear', 'genres', 'averageRating', 'numVotes', 'title']]

# Use the preprocessed df_movies in your existing app
def load_movie_data():
    return df_movies

df_movies = load_movie_data()

# Function to compute cosine similarity and find highest similarity pairs
def compute_movie_similarity(df_movies, description_col, top_n=10):
    combined_descriptions = df_movies[description_col].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_descriptions)
    cosine_sim = cosine_similarity(tfidf_matrix)
    similarity_df = pd.DataFrame(cosine_sim, index=df_movies.index, columns=df_movies.index)
    highest_similarity_pairs = similarity_df.stack().nlargest(top_n)
    return highest_similarity_pairs

# Define colors
maroon_red = "#F05A7E"
golden_yellow = "#FFBE98"
navy_blue = "#125B9A"

# Create GUI
root = ctk.CTk()
root.title("Movie Recommendation System")

frame = ctk.CTkFrame(root, fg_color=navy_blue)
frame.pack(pady=20, padx=20, fill="both", expand=True)

# Add search bar
search_frame = ctk.CTkFrame(frame, fg_color=maroon_red)
search_frame.pack(pady=10, padx=10, fill="x")

search_label = ctk.CTkLabel(search_frame, text="Search Movie:", text_color=golden_yellow)
search_label.pack(side="left", padx=5)

search_entry = ctk.CTkEntry(search_frame)
search_entry.pack(side="left", fill="x", expand=True, padx=5)

search_button = ctk.CTkButton(search_frame, text="Search", command=lambda: search_movie(df_movies, search_entry.get()), fg_color=golden_yellow, text_color=maroon_red)
search_button.pack(side="left", padx=5)

# Add text widget to display results
text_widget = ctk.CTkTextbox(frame, wrap=tk.WORD, fg_color=maroon_red, text_color=golden_yellow)
text_widget.pack(pady=20, padx=20, fill="both", expand=True)

# Function to display similar movies in the GUI
def display_recommendations(df_movies, similarity_pairs):
    text_widget.delete(1.0, tk.END)
    for (i, j), similarity in similarity_pairs.items():
        text = f"Title: {df_movies.loc[j, 'Title']}\n"
        text += f"Plot: {df_movies.loc[j, 'Plot Summary']}\n"
        text += f"Similarity: {round(similarity*100, 0)}%\n{'-'*50}\n"
        text_widget.insert(tk.END, text)

# Set up the main application window with larger dimensions
root = ctk.CTk()
root.geometry("600x400")  # Adjust the window size

# Define a larger font size for the widgets
large_font = ("Arial", 16)

# Create the search entry with a larger font and padding
search_entry = ctk.CTkEntry(root, font=large_font, width=400, height=40)
search_entry.pack(pady=20)

# Create a Text widget with a larger font to display the results
result_text = ctk.CTkTextbox(root, font=large_font, width=500, height=200)
result_text.pack(pady=20)

# Function to search movies and display results in the Text widget
def search_movie(df_movies, query, result_widget):
    # Drop rows with NaN in 'primaryTitle' column
    df_movies = df_movies.dropna(subset=['primaryTitle'])

    # Filter the DataFrame based on the search query
    filtered_movies = df_movies[df_movies['primaryTitle'].str.contains(query, case=False)]

    # Clear the result widget
    result_widget.delete("1.0", tk.END)

    # Insert the filtered results into the result widget
    for _, row in filtered_movies.iterrows():
        result_widget.insert(tk.END, f"{row['primaryTitle']} ({row['startYear']})\n")

# Create a search button with a larger font and padding
search_button = ctk.CTkButton(root, text="Search", command=lambda: search_movie(df_movies, search_entry.get(), result_text), font=large_font, width=200, height=50)
search_button.pack(pady=20)

root.mainloop()

