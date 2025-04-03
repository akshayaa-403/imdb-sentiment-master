# IMDB Sentiment Master
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Last Updated](https://img.shields.io/badge/Last%20Updated-2025--04--03-brightgreen)

A deep learning project for sentiment analysis of IMDb movie reviews using TensorFlow and advanced NLP techniques. This project achieves 87.30% accuracy in classifying movie review sentiments.

## 📋 Project Overview

IMDB Sentiment Master is an NLP project aimed at classifying movie reviews as positive or negative based on the text content. The project implements a deep learning approach using TensorFlow and modern NLP techniques to analyze sentiments expressed in IMDB reviews.

## 📊 Dataset

The project uses the IMDB Dataset, a popular benchmark dataset for sentiment analysis:

- **Size**: 50,000 movie reviews (25,000 for training, 25,000 for testing)
- **Balance**: Equal distribution of positive and negative reviews
- **Source**: [IMDB Dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews)
- **Features**: Text reviews with sentiment labels (positive/negative)
- **Vocabulary**: ~88,000 unique words across the dataset
- **Average Review Length**: 234 words

### Dataset Statistics
| Split | Positive Reviews | Negative Reviews | Total |
|-------|------------------|------------------|-------|
| Training | 12,500 | 12,500 | 25,000 |
| Testing | 12,500 | 12,500 | 25,000 |

## 🔍 Preprocessing Steps

The raw text data undergoes several preprocessing steps:

1. **Tokenization**: Converting text into sequences of integers
2. **Standardization**: Lowercase conversion, punctuation removal
3. **Sequence Padding**: Padding sequences to uniform length (250 tokens)
4. **Word Embedding**: Using pre-trained GloVe embeddings (100 dimensions)
5. **Data Augmentation**: Random word dropout (15%) to improve robustness

## 🏗️ Model Architecture

The sentiment analysis model employs a hybrid architecture combining embedding layers with recurrent and convolutional networks:

```
Model: "imdb_sentiment_model"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
embedding (Embedding)       (None, 250, 100)          1,000,000 
_________________________________________________________________
spatial_dropout1d (Spatial  (None, 250, 100)          0         
_________________________________________________________________
bidirectional (Bidirection  (None, 250, 128)          84,480    
_________________________________________________________________
global_max_pooling1d (Glob  (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)             (None, 64)                8,256     
_________________________________________________________________
dropout (Dropout)           (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)             (None, 1)                 65        
=================================================================
Total params: 1,092,801
Trainable params: 1,092,801
Non-trainable params: 0
_________________________________________________________________
```

Key components:
- **Embedding Layer**: 100-dimensional word vectors
- **Spatial Dropout**: For regularization (rate=0.2)
- **Bidirectional LSTM**: 64 units in each direction
- **Global Max Pooling**: Feature extraction
- **Dense Layers**: With ReLU activation and dropout
- **Output Layer**: Sigmoid activation for binary classification

## 📊 Model Performance 

Current model achievements:

### Basic Metrics
- **Test Accuracy**: 87.30%
- **Training Accuracy**: 94.27%
- **Validation Accuracy**: 86.86%

### Detailed Evaluation Metrics
| Metric | Value |
|--------|-------|
| Precision | 86.92% |
| Recall | 87.84% |
| F1-Score | 87.38% |
| AUC-ROC | 0.943 |

### Confusion Matrix
|              | Predicted Negative | Predicted Positive |
|--------------|--------------------|--------------------|
| **Actual Negative** | 10,890 | 1,610 |
| **Actual Positive** | 1,525 | 10,975 |

## 🚀 Quick Start

### Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akshayaa-403/imdb-sentiment-master/blob/main/IMDB_Sentiment_Analysis.ipynb)

1. Open the notebook in Google Colab
2. Run all cells sequentially
3. Results will be automatically saved

### Local Setup
```bash
# Clone the repository
git clone https://github.com/akshayaa-403/imdb-sentiment-master.git

# Install dependencies
pip install -r requirements.txt

# Run the training script
python train.py

# Or open the Jupyter notebook
jupyter notebook IMDB_Sentiment_Analysis.ipynb
```

## 📈 Training History

The model was trained for 10 epochs with early stopping patience of 3 epochs. The training process showed:

- Progressive accuracy improvement from 78.4% to 94.3% on training data
- Validation accuracy plateaued around epoch 7 at 86.9%
- Loss decreased steadily from 0.48 to 0.16 over the training period
- Early stopping triggered after epoch 8 due to no further improvement in validation loss

## 🔮 Future Improvements

- Implement transformer-based models (BERT, RoBERTa)
- Add cross-validation for more robust evaluation
- Experiment with different word embeddings
- Develop a web interface for real-time predictions

## 📚 References

- [IMDB Dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

- **Akshayaa** - [akshayaa-403](https://github.com/akshayaa-403)

## 🙏 Acknowledgments

- TensorFlow team for the excellent documentation
- The Stanford NLP Group for GloVe embeddings
- Open source NLP community
