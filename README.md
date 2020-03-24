# Covid19_Search_Tool
[Active colab notebook](https://colab.research.google.com/drive/1aFxUJgP1GeMqqw3bUDQIzoYIaYHWKCAr) : Resources for working with CORD19 (Novel Coronovirus 2019) NLP dataset - 

## Getting started
- Visit [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) and download the data (requires Kaggle account)
- Clone this [repository](https://github.com/mrubash1/Covid19_Search_Tool), move the data to Covid19_Search_Tool/data, and unzip the files
- build the attached conda environment
```bash
conda create --name cord19 python=3.6.9
source activate cord19
pip install -r requirements.txt
~/.profile
```
- Dowload the NLTK packages for text processing and search
```bash
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet
```
- Downloading the BERT model by going to Covid_Search_Tool/models
```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
pip install bert-serving-server==1.10 --no-deps
rm uncased_L-12_H-768_A-12.zip
 ```   



## Interactive visualization of COVID-19 related academic articles
![Alt text](img/CORD19_Bert_Embeddings_6000_articles_in_top_journals.png?raw=true "CORD19_Bert_Embeddings_6000_articles_in_top_journals.png")
**TSNE Visualization of COVID-19 related academic articles**
- Color encodes journal
- BERT sentance embeddings are article abstracts
- Using standard BERT pre-trained model (no retraining yet)
- 6200 total articles

### Custom CORD19 NLP Search engine
![Alt text](img/CORD19_nlp_search_engine.png?raw=true "CORD19_nlp_search_engine")
- BM25 natural language search engine
- Data Processing
    1. Remove duplicate articles
    2. Remove (or annotate) non-academic articles (TODO)
- NLP Preprocessing
    1. Remove punctuations and special characters
    2. Convert to lowercase
    3. Tokenize into individual tokens (words mostly)
    4. Remove stopwords like (and, to))
    5. Lemmatize
- [Thanks DwightGunning for the great starting point here!](https://colab.research.google.com/drive/1aFxUJgP1GeMqqw3bUDQIzoYIaYHWKCAr)

### Plan of action
- Topic modeling with LDA @Rachael Creager 
- NLU feature engineering with TF-IDF @Maryana Alegro 
- NLU feature engineering with BERT @Matt rubashkin
- Feature engineering with metadata
- Making an embedding search space via concatenating the TOPIC, NLU and metadata vectors @Kevin Li
- Then Creating a cosine sim search engine that creates the same datatype as the above vector
- Streamlit app that has search bar, and a way to visualize article information (Mike Lo)

### Current work based on:
- [BM 25 Search Engine by DwightGunning](https://colab.research.google.com/drive/1aFxUJgP1GeMqqw3bUDQIzoYIaYHWKCAr)
- [Building a search engine with BERT and Tensorflow](https://colab.research.google.com/drive/1ra7zPFnB2nWtoAc0U5bLp0rWuPWb6vu4)
