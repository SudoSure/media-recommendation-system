# Media Recommendation System (In-Progress)

A media recommendation system that allows users to search for movies, anime, and TV shows and view recommendations based on cosine similarity of descriptions. The system is built using Python with the `pandas`, `numpy`, `sklearn`, and `customtkinter` libraries.

## Features

- **Load and preprocess IMDB data**: Load movie, TV show, and anime data from IMDB, filter out non-relevant content, and prepare it for analysis.
- **Media similarity computation**: Compute similarity between media items based on their descriptions using TF-IDF vectorization and cosine similarity.
- **GUI interface**: User-friendly graphical interface to search for media items and view recommendations.
- **Search functionality**: Search for media items by title and display results in a text widget.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/media-recommendation-system.git
    cd media-recommendation-system
    ```

2. **Install dependencies:**

    It is recommended to create a virtual environment before installing the dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Download IMDB data files:**

    Ensure you have the required IMDB data files (`title.basics.tsv.gz`, `title.ratings.tsv.gz`, `title.akas.tsv.gz`) in the project directory.

## Usage

1. **Run the application:**

    ```bash
    python main.py
    ```

2. **Use the GUI:**

    - Enter a media title in the search bar.
    - Click the "Search" button to find and display media matching the query.
    - View recommendations based on similarity in the text widget.

## Code Overview

### `load_imdb_data(filename)`

Loads IMDB data from a gzip-compressed TSV file.

### `load_media_data()`

Returns the preprocessed DataFrame with media data, including movies, TV shows, and anime.

### `compute_media_similarity(df_media, description_col, top_n=10)`

Computes cosine similarity between media items based on a specified description column and returns the top `n` similar pairs.

### GUI Components

- **Search Bar**: Allows users to input a media title to search for.
- **Search Button**: Triggers the search operation and displays results.
- **Results Display**: Shows media titles and similarity scores.

## Requirements

- Python 3.x
- `pandas`
- `numpy`
- `scikit-learn`
- `customtkinter`
- `tkinter`

You can create a `requirements.txt` file to manage dependencies:

```txt
pandas
numpy
scikit-learn
customtkinter
tkinter
