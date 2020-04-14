---
layout: prediction_post
published: True
title: Jiasaw Multilingual Toxic Comment Classification-Starter Blog
---

This project is an ongoing Kaggle Competition. It lives at [toxic comment classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification). It is a natural language processing (NLP) task. I chose this topic as the final project because NLP is a very hot topic nowadays and I am new to this area. I hope to take advantages of this opportunity to learn more about deep learning targeted towards the state-of-art application in NLP. 

In this starter blog, I will walk you through the overview of the competition, exploratory data analysis, and  basics of language models for this project. The outline will be as follows.

<div style="font-size:75%; background-color:#eee; border: 1px solid #bbb; display: table; padding: 7px" markdown="1">

<div style="text-align:center" markdown="1">  

**Contents**

</div>

* **[Part 1: Introduction](#part-1-introduction-and-background)**
  * Background & Motivation
  * Description of The Competition
  * Evaluation Metrics and Submission Requirements
* **[Part 2: Data Exploration](#part-2-eda)**
  * Dataset
  * Preprocessing
  * EDA
* **[Part 3: Basics of Language Models](#part-3-basics-of-language-models)**
  * What is a Language Model?
  * Word Embeddings
  * Attention

</div>

## <a href="#part-1-introduction-and-background" name="part-1-introduction-and-background">Part 1: Introduction </a>

### Background & Motivation
Thanks to the rapid development of deeping learning techniques and computational hardwares, NLP has been gaining its momentum in the past two decades. As believed by machine learning experts, NLP is experiencing a boom in the short-term future, same as computer vision once did. The popularity of it brought a great amount of investment. Recently Kaggle released two NLP competitions ([tweet sentiment extraction](https://www.kaggle.com/c/tweet-sentiment-extraction) and [comment toxicity analysis](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)). Of focus here is the second one because it is based off two previous Kaggle competitions regarding the same topic ([2018 toxicity](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) and [2019 toxicity](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)). For the very first competion, contestants are challenged to buld multi-headed models to recognize toxicity and several subtypes of toxicity. *Toxicity is defined as anything rude, disrespectful or other wise likely to make someone leave a discussion*. The 2019 Challenges asks Kagglers to work across a diverse range of conversations. The main purpose of this final project is to understand the basics of deep learning techniques applied to NLP. So it would be more doable to work on a project in such a limited time for which there exist many established references/documents. 

 
### Description of The Competition
Taking advantage of Kaggle's TPU support, this competition aims to build multilingual models with English-only training data. The model will be tested on Wikipedia talk page comments in several different languages. It is supported by [The Conversation AI team], funded by [Jiasaw](https://jigsaw.google.com/) and Google. Its mission is *to create future-defining research and technology to keep our world safer*. 

### Evaluation Metrics and Submission Requirements
Basically it is a classification problem. The model performance is evaluated by the [area under the ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predictions and the observations.

The submission file consists of two columns. The first column indicates the comment `id` and the second one is the probability for the `toxicity` variable. Following is an example submission file.

<html>
<head>
<style>
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
th, td {
  padding: 5px;
}
th {
  text-align: left;
}
</style>
</head>
<body>

<table style="width:50%">
  <tr>
    <th>id</th>
    <th>toxic</th>
  </tr>
  <tr>
    <td>0</td>
    <td>0.3</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0.7</td>
  </tr>
  <tr>
    <td>2</td>
    <td>0.9</td>
  </tr>
</table>
</body>
</html>

In addition to the well defined metrics evaluted on the given testing set. We might also want to futher apply the language model to additional applications. For example

* As mentioned before, there is another NLP competition on Kaggle, which challenges contestants to analyze the tweet sentiment. Basically there are three types of sentiment, including *neural*, *negative* and *positive*. 

* Another possible application is to scrape comments from some social media, say "reddit", and predict whether the comment will receive upvote, downvote or be removed.

## <a href="#part-2-eda" name="part-2-eda">Part 2: Data Exploration </a>

### Dataset
Following is the list of the datasets we have for this project. The primary data is the `comment_text` column which contains the text of comment to be classified as toxic or non-toxic (0...1 in the `toxic` column). The trainingset's comments are mostly written in English whereas the validation and testing sets' comments are composed of multiple non-English languages. A detailed explanation of the dataset can be found on the [competition webpage](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/data)

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/datasets.png"/>
  <br />
</div>

Below shows the header of the training set, validation set and testing set. There are mainly four columns for all datasets, in which `id` is the identifier, `commen_text` is the text of comment, `lang` is the language of the comment, and `toxic` is whether or not the comment is toxic. In the training set, we can see 5 additional columns which represent the subtypes of toxic comment. Moreover, we do not have the `toxic` column in the testing set.

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/train_header.png"/>
  <br />
  <figcaption>Top 5 rows of the training set</figcaption>
</div>

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/validation_header.png"/>
  <br />
  <figcaption>Top 5 rows of the validation set</figcaption>
</div>

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/test_header.png"/>
  <br />
  <figcaption>Top 5 rows the testing set</figcaption>
</div>

### Preprocessing
We can do a few data preprocessing steps before feeding the data into a language model. 

- Clean up the comment texts by dropping redundant information, such as usernames, emails, hyperlinks and line breakers.

- Remove unnecessary columns in the trainingset such as the subtypes of toxic because the target is only `toxic`.

- Tokenize the words, which can also be taken as a step of setting up a model.

### EDA

First we take an overview of the comments in the training set. 

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/common_words.png"/>
  <br />
  <figcaption>Wordclouds of the comment texts</figcaption>
</div>

We can see that the most common words include "Wikipedia", "article", "will" and "see". Aggressive and disrespectful words seems to occur less often.

The figure below shows the distribution of the length of the comment texts. One can see that the distribution if right-skewed and peaked at around a position of $$13$$ words. 

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/comments_length.png"/>
  <br />
  <figcaption>Comment length distribution</figcaption>
</div>

This bar plot indicates that the balance of the dataset is around $$21384/(21384+202165) \approx 90\%$$. 

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/balance.png"/>
  <br />
  <figcaption>Counts of the toxic and non-toxic comments</figcaption>
</div>

Lastly we summarize the common words in the toxic comments in another worldclouds plot. *Disclaimer: The following figure contains text that may be considered profane, vulgar, or offensive.* 

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/toxic_common_words.png" />
</div>

Obviously, toxic comments use more insulting or hateful words such as "f\*\*k". 

## <a href="#part-3-basics-of-language-models" name="part-3-basics-of-language-models">Part 3: Basics of Language Models </a>

### What is a Language Model?
A language model is basically a machine learning model that looks at part of a sentence and is able to predict the next one, such as next word recommendation for cellphone keyboard typing. 

<div class="img-div" markdown="0" style="text-align:center">
  <image src="http://jalammar.github.io/images/word2vec/swiftkey-keyboard.png"/>
  <br />
</div>

Statistically, a language model is a probability distribution over sequence of words. Most language models rely on the basic assumption that the probability of a word only depends on the previous $$n$$ words, which is known as the $$n$$-gram model. Langugae models are useful in many scenarios such speech recognition, parsing and information retrieval. For more explanation, please refer to the wiki page for [language models](https://en.wikipedia.org/wiki/Language_model). 

### Word Embeddings
Word embedding is a type of word respresentation that allows words with similar meaning to have a similar representation. It is a groundbreaking progress for developing high-performance deep learning models for NLP. The intuitive approach to word representation is the **one-hot** encoding. To represent each word, we create a zero vector with length equal to the vocabulary. Then one is placed in the index that corresponds to the word. In that sense, we will create a sparse vector. An alternative approach is to encode each word with a unique number so that the resulting vector is short and dense. However, the way how each word is encoded is arbitrary, and we do not know the relationship between the words. Here comes the technique of **word embeddings**. In this scenario, we do not have to specify the encoding by hand. Instead of manually defining the embedding vector, the values of the vector are trained in the same way a model learns weights of a dense layer. A high-dimensional embedding can capture fine relationships between words. More articles about word embedding can be found in the following readings.

- [What are word embeddings?](https://machinelearningmastery.com/what-are-word-embeddings/)

- [Word embeddings in Tensorflow](https://www.tensorflow.org/tutorials/text/word_embeddings)

- [NLP: Everything about Embeddings](https://medium.com/@b.terryjack/nlp-everything-about-word-embeddings-9ea21f51ccfe)

### Attention

The key idea of Attention is to focus on the most relevant parts of the input sequence as needed. It provides a direct path to the inputs. So it also alleviates the vanishing gradient issue. This significantly improves the model performance when confronting with long sentence analysis. 

For a typical language model, it is composed of an encoder and a decoder.
The encoder processes each item in the input sequence, and then compile the transformed information into a vector. After processing the entire input sequence, the encoder send the context to the decoder for the next step. Both the encoder and decoder are intrinsically recurrent nueral networks (RNN) which processes the input vector and previous hidden state, and produces the next-step hidden state and output at that time step. 

At a high level of abstraction, an attention model differs in two main ways. Firstly, instead of passing only the last hidden state at the encoder side, the attention model holds all the hidden states and passes all hidden state to the decoder. Secondly, in the decoder side it does one more step before calculating its output. The basic idea is that each hidden state produced at the encoder side is associated with a certain word in the input sequence, thus we can assign a score to each hidden state and use that to amplify the word with high score and drown out words with low scores. A illustrative and comprehensive tutorial of an attention model can be found in the blog [visualizing a neural machine translation model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/). Other useful links are also attached at below.

- [Attention and its Different Forms](https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc)

- [NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

## Acknowledgements

- [EDA and Modelling Kernel](https://www.kaggle.com/tarunpaparaju/jigsaw-multilingual-toxicity-eda-models) ~ by Tarun Paparaju

- [Illustrative Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) ~ by Jay Alammer

- [Polyglot](https://pypi.org/project/polyglot/) ~ by aboSamoor

- [github blog template](https://github.com/barryclark/jekyll-now) ~ by Barry Clark
