# Movie Genre Classification using Machine Learning:

Movie Genre Classification is a machine learning-based text classification system that predicts a movie's genre from its plot description using NLP techniques and classification algorithms.


## Project Overview:

This project demonstrates the application of text classification techniques in machine learning and serves as an excellent beginner-to-intermediate NLP project.


## Objectives:

- Classify movies into different genres based on plot descriptions.
- Apply text preprocessing and feature extraction techniques.
- Build and evaluate a machine learning classification model.
- Understand NLP workflows using Python and Scikit-learn.


## Features:

- Text-based movie genre prediction.
- TF-IDF feature extraction.
- Multinomial Naive Bayes classifier.
- Model evaluation using accuracy score.
- Classification report generation.
- Confusion matrix visualization.
- Interactive genre prediction from user input.


## Technologies Used:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn


## Machine Learning Workflow:

1. Data Collection
2. Data Preprocessing
3. Text Feature Extraction using TF-IDF
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Genre Prediction


## Dataset:

The project uses movie plot descriptions and corresponding genres.


### Dataset Attributes:

| Feature | Description |
|----------|-------------|
| Plot | Movie storyline or summary |
| Genre | Movie category label |

Example:

| Plot | Genre |
|--------|--------|
| A superhero saves the world from dangerous villains | Action |
| Two people fall in love despite family conflicts | Romance |
| A detective investigates a mysterious murder case | Thriller |


## Algorithms Used:

### TF-IDF Vectorizer:

TF-IDF (Term Frequency-Inverse Document Frequency) converts textual movie plots into numerical feature vectors suitable for machine learning.

### Multinomial Naive Bayes:

A probabilistic classification algorithm widely used for text classification tasks due to its simplicity and efficiency.


## Project Structure:

Movie_Genre_Classification/

├── movie_genre_classification.py

├── README.md

├── dataset.csv (optional)

├── outputs/

│   ├── confusion_matrix.png

│   └── results.txt

└── requirements.txt


## Install Required Libraries:

pip install pandas numpy matplotlib scikit-learn


### Clone the Repository:

git clone https://github.com/yourusername/movie-genre-classification.git
cd movie-genre-classification


## Results:

The classifier successfully predicts movie genres based on plot summaries. Performance can be improved by training on larger datasets and using advanced NLP models.


## Conclusion:

Movie Genre Classification demonstrates how machine learning and NLP techniques can automatically identify movie genres from textual descriptions.
