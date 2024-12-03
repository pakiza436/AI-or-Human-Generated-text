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



## Conclusion
This work showcases the potential of combining machine learning, deep learning, and NLP techniques for AI-based text analysis and classification. It provides a foundation for further advancements in distinguishing AI-generated text from human-written content.
