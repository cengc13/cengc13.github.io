---
layout: prediction_post
published: True
title: Jigsaw Multilingual Toxic Comment Classification-Start Blog
---

This is the first of three blogs documenting my entry into the [toxic comment classification kaggle competition](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification). It is a natural language processing (NLP) task. I chose this topic as the final project because NLP is a very hot topic nowadays and I am new to this area. I hope to take advantages of this opportunity to learn more about deep learning targeted towards the state-of-art application in NLP. 

In the first blog, I walk you through an overview of the competition, the exploratory data analysis, and  the basics of language models for this project.

<!-- <center><img src="https://i.imgur.com/4WNesOq.png" width="400px"></center> -->

<div class="img-div" markdown="0" style="text-align:center">
  <image src="https://i.imgur.com/4WNesOq.png" width="400px" />
  <br />
  <figcaption>Competition Logo. Source:
    <a href="https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge">Logo</a></figcaption>
</div>

<!--more-->

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
Taking advantage of Kaggle's TPU support, this competition aims to build multilingual models with English-only training data. The model will be tested on Wikipedia talk page comments in several different languages. It is supported by The Conversation AI team, which is funded by [Jiasaw](https://jigsaw.google.com/) and Google. 

### Evaluation Metrics and Submission Requirements
Basically it is a classification problem. The model performance is evaluated by the [area under the ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predictions and the observations.

The submission file consists of two columns. The first column indicates the comment `id` and the second one is the probability for the `toxicity` variable. Following is a sample submission file.

<table class="features-table">
  <tr>
    <th class="mdc-text-light-green-600">
    id
    </th>
    <th class="mdc-text-purple-600">
    toxic
    </th>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      0
    </td>
    <td class="mdc-bg-purple-50">
      0.3
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      1
    </td>
    <td class="mdc-bg-purple-50">
      0.7
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
     2
    </td>
    <td class="mdc-bg-purple-50">
      0.9
    </td>
  </tr>
</table>

In addition to the well defined metrics evaluted on the given testing set. We might also want to futher apply the language model to additional applications. For example,

* As mentioned before, there is another NLP competition on Kaggle, which challenges contestants to analyze the tweet sentiment. Basically there are three types of sentiment, including *neural*, *negative* and *positive*. 

* Another possible application is to scrape comments from some social media, say "reddit", and predict whether the comment will receive upvote, downvote or be removed.

## <a href="#part-2-eda" name="part-2-eda">Part 2: Data Exploration </a>

### Dataset
Following is the list of the datasets we have for this project. The primary data is the `comment_text` column which contains the text of comment to be classified as toxic or non-toxic (0...1 in the `toxic` column). The trainingset's comments are mostly written in English whereas the validation and testing sets' comments are composed of multiple non-English languages. A detailed explanation of the dataset can be found on the [competition webpage](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/data)

<!-- <div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/datasets.png"/>
  <br />
</div> -->
<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/train_header.png" width="800px"/>
  <br />
  <figcaption>Top 5 rows of the training set</figcaption>
</div>

<br/><br/> 

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/validation_header.png" width="800px"/>
  <br />
  <figcaption>Top 5 rows of the validation set</figcaption>
</div>

Below shows the five top rows of the training set, validation set and testing set. There are mainly four columns for all datasets, in which `id` is the identifier, `commen_text` is the text of comment, `lang` is the language of the comment, and `toxic` is whether or not the comment is toxic. In the training set, we can see 5 additional columns which represent the subtypes of toxic comment. Moreover, we do not have the `toxic` column in the testing set.


<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/test_header.png" width="800px"/>
  <br />
  <figcaption>Top 5 rows the testing set</figcaption>
</div>

As mentioned before, most comments in the training set are in English while most comments in validation and testing set are in Non-English, including Spanish, French, Turkish and portuguese etc. The number for all types of languages in validation and test set are summarized at below.

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/validation_languages.png" width="800px"/>
  <br />
  <figcaption>Language counts in the validation set</figcaption>
</div>

<br/><br/> 

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/test_languages.png" width="800px"/>
  <br />
  <figcaption>Language counts in the test set</figcaption>
</div>

### Preprocessing
We can do a few data preprocessing steps before feeding the data into a language model. 

- Clean up the comment texts by dropping redundant information, such as usernames, emails, hyperlinks and line breakers.

- Remove unnecessary columns in the trainingset such as the subtypes of toxicity because the target for submission is only the `toxic`.

- Tokenize the words, which can be also considered as a step for building up a model.

### EDA

**Note that** the analysis for "wordcloud" is  inspired by this kernel [EDA and Modeling Kernel](https://www.kaggle.com/tarunpaparaju/jigsaw-multilingual-toxicity-eda-models). 

#### Comment Wordcloud
Firstly we take a look at the comments in the training set. 

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/comment_wordcloud.png" width="800px"/>
</div>

The most common words include "Wikipedia", "article", "will" and "see". 

Another plot in the following shows the wordcloud for common words in the toxic comments. 

*Disclaimer: The following figure contains text that may be considered profane, vulgar, or offensive.* 

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/toxic_wordcloud.png" width="800px"/>
</div>

As expected, there exist more insulting or hateful words, such as "die" and "pig". 

#### Histograms of number of words and sentences in all comments  

The figure below shows the distribution for number of words in all comments. One can see that the distribution is right-skewed, and it is peaked at 13 words per comment.

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/comment_words.png" width="800px"/>
  <br />
  <figcaption>Distribution of number of words in all comments</figcaption>
</div>

#### Histogram of number of sentences in all comments 

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/comment_sentences.png" width="800px"/>
  <br />
  <figcaption>Distribution of number of sentences in all comments</figcaption>
</div>

The distribution for number of sentences is also right skewed, and 

#### Balance of training set

This bar plot indicates that the balance of the dataset is about 90%. The dataset is hence highly unbalanced.

<div class="img-div" markdown="0" style="text-align:center">
  <image src="/images/starter-blog/balance.png" width="800px"/>
  <br />
  <figcaption>Counts of the toxic and non-toxic comments</figcaption>
</div>

## <a href="#part-3-basics-of-language-models" name="part-3-basics-of-language-models">Part 3: Basics of Language Models </a>

### What is a Language Model?
A language model is basically a machine learning model that looks at part of a sentence and is able to predict the next one, such as next word recommendation for cellphone keyboard typing. 

Statistically, a language model is a probability distribution over sequence of words. Most language models rely on the basic assumption that the probability of a word only depends on the previous *n* words, which is known as the *n*-gram model. Langugae models are useful in many scenarios such speech recognition, parsing and information retrieval. Please refer to the [wiki  page](https://en.wikipedia.org/wiki/Language_model) for more information. 

### Word Embeddings
Word embedding is a type of word respresentation that allows words with similar meaning to have a similar representation. It is a groundbreaking progress for developing high-performance deep learning models for NLP. The intuitive approach to word representation is the **one-hot** encoding. To represent each word, we create a zero vector with length equal to the vocabulary. Then one is placed in the index that corresponds to the word. In that sense, we will create a sparse vector. An alternative approach is to encode each word with a unique number so that the resulting vector is short and dense. However, the way how each word is encoded is arbitrary, and we do not know the relationship between the words. Here comes the technique of **word embeddings**. In this scenario, we do not have to specify the encoding by hand. Instead of manually defining the embedding vector, the values of the vector are trained in the same way a model learns weights of a dense layer. A high-dimensional embedding can capture fine relationships between words. 

### Attention

The key idea of Attention is to focus on the most relevant parts of the input sequence as needed. It provides a direct path to the inputs. So it also alleviates the vanishing gradient issue. This significantly improves the model performance when confronting with long sentence analysis. 

For a typical language model, it is composed of an encoder and a decoder.
The encoder processes each item in the input sequence, and then compile the transformed information into a vector. After processing the entire input sequence, the encoder send the context to the decoder for the next step. Both the encoder and decoder are intrinsically recurrent nueral networks (RNN) which processes the input vector and previous hidden state, and produces the next-step hidden state and output at that time step. 

At a high level of abstraction, an attention model differs in two main ways. Firstly, instead of passing only the last hidden state at the encoder side, the attention model holds all the hidden states and passes all hidden state to the decoder. Secondly, in the decoder side it does one more step before calculating its output. The basic idea is that each hidden state produced at the encoder side is associated with a certain word in the input sequence, thus we can assign a score to each hidden state and use that to amplify the word with high score and drown out words with low scores. A illustrative and comprehensive tutorial of an attention model can be found in the blog [visualizing a neural machine translation model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/). 

## Citations and References

- Tarun Paparaju. (2020, March). *Jigsaw Multilingual Toxicity : EDA + Models*. Retrieved from https://www.kaggle.com/tarunpaparaju/jigsaw-multilingual-toxicity-eda-models  

- Jay Alammer. (2018, May 9). *JVisualizing A Neural Machine Translation Model*. Retrieved from https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/

- Barry Clark. (2016, March). *Build a Jekyll blog in minutes, without touching the command line*. Retrieved from https://github.com/barryclark/jekyll-now

-  Jason Brownlee. (2017, October 11). *What Are Word Embeddings for Text?* Retrieved from https://machinelearningmastery.com/what-are-word-embeddings/

- Mohammed Terry-Jack. (2019, April 21). *NLP: Everything about Embeddings*. Retrieved from https://medium.com/@b.terryjack/nlp-everything-about-word-embeddings-9ea21f51ccfe

- Anusha Lihala. (2019, March 29). *Attention and its Different Forms*. Retrieved from https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc

- Sean Robertson. (2017). *NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION*. Retrieved from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html