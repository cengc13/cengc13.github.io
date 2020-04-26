---
layout: prediction_post
published: True
title: Jigsaw Multilingual Toxic Comment Classification-Midway Blog
---

This blog is the second of the three blogs documenting my entry into [toxic comment classification kaggle competition](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification). In the [first blog](https://cengc13.github.io/final-project-start-blog/), we introduced the dataset, the EDA analysis and some fundamental knowledge about a language model. To move forward, the primary purpose of the next step is to develop the baseline model from scratch. The essential components of a language model will be summarized, such as the tokenizer, the model architecture, and the evaluation metrics. In addition, we will cover some state-of-the-art multilingual models, such as BERT, XLM and XLM-RoBERT.

<center><img src="https://www.topbots.com/wp-content/uploads/2019/02/NLP_feature_image_1600px-1280x640.jpg" width="400px"></center>


<!--more-->

<div style="font-size:75%; background-color:#eee; border: 1px solid #bbb; display: table; padding: 7px" markdown="1">

<div style="text-align:center" markdown="1">  

**Contents**

</div>

* **[Part 1: The Baseline Model](#part-1-baseline-model)**
  * Dataset
  * Tokenizer
  * The Model
* **[Part 2: Cross-lingual Modeling](#part-2-multilingual-models)**
  * BERT and its Variants
  * XLM
  * XLM-RoBERTa

</div>


## <a href="#part-1-baseline-model" name="part-1-baseline-model">Part 1: The Baseline Model </a>

Our goal is to take as input a comment text, and produces either 1(the comment is toxic) or 0 (the comment is non-toxic). It is basically a binary classification problem. The simplest model we can think of is the logistic regression model, for which we need to figure out how to digitalize comments so that we can use logistic regression to predict the probabilities of a comment being toxic. Next we will do a quick overview of the dataset, introduce the concepts of tokenizer, and go over the architecture of a baseline model.

### Dataset: Jigsaw Multilingual Comments

The dataset we will use, as mentioned in the first blog, is from the Kaggle competition [Jigsaw Multilingual Toxic Analysis](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification), which contains the comment texts and its toxicity labels, indicating whether the comment text is disrespectful, rude or insulting. 

<table class="features-table">
  <tr>
    <th class="mdc-text-light-green-600">
    Comment
    </th>
    <th class="mdc-text-purple-600">
    Toxic
    </th>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      This is so cool. It's like, 'would you want your mother to read this??' Really great idea, well done!
    </td>
    <td class="mdc-bg-purple-50">
      0
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      Thank you!! This would make my life a lot less anxiety-inducing. Keep it up, and don't let anyone get in your way!
    </td>
    <td class="mdc-bg-purple-50">
      0
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      This is such an urgent design problem; kudos to you for taking it on. Very impressive!
    </td>
    <td class="mdc-bg-purple-50">
      0
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      haha you guys are a bunch of losers.
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      Is this something I'll be able to install on my site? When will you be releasing it?
    </td>
    <td class="mdc-bg-purple-50">
      0
    </td>
  </tr>
</table>

We can load the dataset with `pandas`. Then we split the dataset to train and test sets in a stratified fashion as the dataset is highly unbalanced.
The splitting ratio is 8:2.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
train = pd.read_csv("./jigsaw-toxic-comment-train.csv")
X, y = train.comment_text, train.toxic
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)
y_train, y_test = y_train.astype(int), y_test.astype(int)
```


### Tokenizer

A tokenizer works as a pipeline. It processes some raw text as input and output encoding. It is usually structured into three steps. Here we will use the example provided in the blog ["A Visual Guide to Using BERT for the First Time"](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/). For instance, if we would like to classify the sentence "“a visually stunning rumination on love", the tokenizer will firstly split the sentences into words with some separator, say whitespace. In the next step, special tokens will be added for sentence classifications for some tokenizers. 

<center><img src="http://jalammar.github.io/images/distilBERT/bert-distilbert-tokenization-1.png" width="400px"></center>


The final step is to replace each token with its numeric id from the embedding table, which is a natural component of a pre-trained model. Then the sentence is ready to be sent for a language model to be processed.

<center><img src="http://jalammar.github.io/images/distilBERT/bert-distilbert-tokenization-2-token-ids.png" width="400px"></center>

For the purpose of demonstration, in the baseline model, we will use a classic tokenization method `TF-IDF`, which is short for "term frequency-inverse document frequency". Basically it counts the number of occurrence of a word in the documents, and then it is offset by the number of documents that contain the word. This tokenization approach is available in the package `sklearn`. 

```python
### Define the vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=2000, min_df=2,
								   max_df=0.95)
### Suppose X_train is a corpus of texts
## Fit the vectorizer
X_train_fitted = tfidf_vectorizer.fit_transform(X_train)
X_test_fitted = tfidf_vectorizer.transform(X_test)
```

In addition, (HUGGING FACE)[https://huggingface.co/] provides a open-source package, named `tokenizer`, where you can find many fast state-of-the-art tokenizers for research and production. For example, to implement a pre-trained DistilBERT tokenizer and model/transformer, you just need two-line codes as follows

```python
import transformers as ppb
# For DistilBERT:
tokenizer_class, pretrained_weights = (ppb.DistilBertTokenizer, 
									   'distilbert-base-uncased')
# load pretrained tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
```

After tokenization, we can build a model and train it with the tokenized comments.

### The Model

We define the simplest binary classification model with logistic regression. 

```python
from sklearn.linear_model import LogisticRegression
# C is a term to control the l2 regularization strength
model_lr = LogisticRegression(C=6.0)
```
If you want to optimize the hyperparameter `C`, you can do a simple grid search.

```python
from sklearn.model_selection import GridSearchCV
parameters = {'C': np.linspace(0.0001, 100, 20)}
grid_search = GridSearchCV(LogisticRegression(), parameters)
grid_search.fit(X_train_fitted, y_train)

print('best parameters: ', grid_search.best_params_)
print('best scrores: ', grid_search.best_score_)
```

We train and evaluate the model by
```python
## training
model_lr.fit(X_train_fitted, y_train)
## prediction on testing set
model_lr.score(X_test_fitted, y_test)
```

Dive right into the [notebook](https://github.com/cengc13/2040FinalProject/blob/master/src/models/logistic_regression.ipynb) or [running it on colab](https://drive.google.com/file/d/1bVBPSKS0JGhOUUaj1yiNmDYRwnFxNsYS/view?usp=sharing). 

## <a href="#part-2-multilingual-models" name="part-2-multilingual-models">Part 2: Cross-lingual Models </a>

