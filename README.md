# AI-Generated vs Human-Written Text Classification

## Project Overview

This project aims to distinguish between AI-generated text and human-written text using the DIAGT dataset from Kaggle. By leveraging natural language processing (NLP) techniques, the project employs a comprehensive pipeline involving both machine learning and deep learning algorithms.

### Key Highlights

- **Machine Learning Models**:
  - Logistic Regression
  - XGBoost
  - Random Forest
  - Support Vector Machine (SVM)
  - Naïve Bayes
  - Ensemble Models

- **Deep Learning Models**:
  - Multilayer Perceptron (MLP)
  - Gated Recurrent Units (GRU)
  - Long Short-Term Memory (LSTM)
  - Convolutional Neural Network (CNN)
  - Hybrid CNN-LSTM model
# Dataset

## Name: DIAGT Dataset  

### Available Link:  
[DIAGT External Dataset](https://www.kaggle.com/datasets/xxycbadpanda/daigt-external-dataset-v1)  

### Description:  
The DIAGT dataset, obtained from Kaggle, contains two main categories for classification:  
- **Text Column**: Contains textual data to be classified.  
- **Labels Column**: Defines the source of the text using binary values:  
  - `1`: AI-generated text  
  - `0`: Human-written text  

The dataset serves as the foundation for distinguishing between AI-generated and human-written text using advanced natural language processing (NLP) and machine learning techniques.  

### Background:  
The rapid advancement of AI, particularly in natural language processing, has led to the emergence of large language models capable of generating human-like text. The DIAGT dataset provides a platform to analyze and classify text origin, contributing to the broader study of AI-generated content detection.  

This dataset is pivotal in exploring methodologies that combine machine learning and deep learning algorithms to detect patterns in text classification and understand the distinction between AI and human text creation.  

### Focus:  
This project specifically focuses on using the DIAGT dataset for text classification to identify whether a given piece of text is AI-generated or written by a human.  

## Setup Instructions  
To run the code for this project, use **Google Colab GPU** for faster computation.  
# Repository Structure

- **DIAGT_Text_Classification.ipynb**: The primary notebook for text classification and analysis.
- **diagt_dataset.rar**: Dataset file used for training and evaluation.
- **README.md**: Project documentation.

---

## Getting Started

### Clone the Repository
First, clone this repository to your local machine:

```bash
git clone  https://github.com/pakiza436/AI-or-Human-Generated-text.git
## Running on Google Colab

You can run the notebook directly on Google Colab. Follow these steps:

1. **Open Google Colab**: Go to [Google Colab](https://colab.research.google.com/).

2. **Upload the Notebook**: Upload the `DIAGT_Text_Classification.ipynb` notebook or use the Colab link from the repository.

3. **Install Required Libraries**: Install the necessary dependencies by running the following commands in a code cell:

  
### Required Libraries:  
Install and import the following libraries for text processing, feature extraction, model training, and evaluation:  

```python
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import spacy
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation

import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
## 3. Dataset (diagt_dataset.rar)

- **Google Colab**: Upload the dataset manually or link it from Google Drive.

---

## 4. Key Libraries Used

- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **NLTK**: Natural language processing toolkit for tokenization, stemming, and stopword removal.
- **Scikit-learn**: For machine learning model training and evaluation.
- **Gensim**: For advanced topic modeling and word embeddings.
- **SpaCy**: For NLP tasks like lemmatization and POS tagging.
- **TextBlob**: For sentiment analysis.

---

## 5. Run the Notebook

- Launch Google Colab and open the `DIAGT_Text_Classification.ipynb` file.

---

## 6. Training the Model

The notebook allows you to train and evaluate various models, including:

### Machine Learning Models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- Naïve Bayes

### Deep Learning Models:
- Multilayer Perceptron (MLP)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)
- Convolutional Neural Network (CNN)
- Hybrid CNN-LSTM models

**Note**: Make sure to adjust the `train_dataloader` and `val_dataloader` sections to fit your dataset.

---

## Notes

1. **NLTK Resources**: The project uses NLTK resources like `punkt`, `stopwords`, and `wordnet`. Ensure they are downloaded during the setup phase.

2. **TensorFlow Models**: TensorFlow is used for neural network-based text classification.

4. **Google Colab GPU**: Enable GPU for faster training by navigating to **Runtime > Change runtime type > Hardware accelerator > GPU**.

---

## Preprocessing

The preprocessing steps in this project are designed to clean and prepare the text data for model training. The steps include:

- **Lowercasing**: Converts all text to lowercase.
- **Removal of URLs**: Removes any URLs from the text using regular expressions.
- **Removal of Email Addresses**: Strips email addresses from the text.
- **Removal of HTML Tags**: Eliminates any HTML tags from the text.
- **Removal of Special Characters and Digits**: Filters out non-alphabetical characters and digits.
- **Removal of Punctuation**: Removes all punctuation marks from the text.
- **Tokenization**: Breaks the text into individual words.
- **Stop Words Removal**: Removes common stop words (e.g., "the", "is", "in") using the NLTK stopwords list.
- **Stemming**: Applies the Porter Stemmer to reduce words to their base form (e.g., "running" -> "run").
- **Lemmatization**: Uses the WordNet Lemmatizer to convert words to their root form (e.g., "better" -> "good").

---

## Features Applied

### For Shallow Machine Learning Models:
- **TF-IDF**
- **N-grams**
- **Topic Modeling**
- **Sentiment Analysis**

---

## Performance Evaluation Measures

The models are evaluated based on the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

These metrics provide a comprehensive assessment of the models' performance.


## Conclusion
This work showcases the potential of combining machine learning, deep learning, and NLP techniques for AI-based text analysis and classification. It provides a foundation for further advancements in distinguishing AI-generated text from human-written content.
