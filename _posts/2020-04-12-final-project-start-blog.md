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
* **[Part 2: EDA](#part-2-eda)**
  * Dataset
  * Preprocessing
  * EDA
* **[Part 3: Basics of Language Models](#part-3-basics-of-language-models)**
  * What is a Language Model?
  * Word Embeddings
  * Self Attention
  <!-- * BERT -->

</div>

## <a href="#part-1-introduction-and-background" name="part-1-introduction-and-background">Part 1: Introduction </a>

### Background & Motivation
Thanks to the rapid development of deeping learning techniques and computational hardwares, NLP has been gaining its momentum in the past two decades. As believed by machine learning experts, NLP is experiencing a boom in the short-term future, same as computer vision once did. The popularity of it brought a great amount of investment. Recently Kaggle released two NLP competitions ([tweet sentiment extraction](https://www.kaggle.com/c/tweet-sentiment-extraction) and [comment toxicity analysis](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)). Of focus here is the second one because it is based off two previous Kaggle competitions regarding the same topic ([2018 toxicity](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) and [2019 toxicity](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)). For the very first competion, contestants are challenged to buld multi-headed models to recognize toxicity and several subtypes of toxicity. *Toxicity is defined as anything rude, disrespectful or other wise likely to make someone leave a discussion*. The 2019 Challenges asks Kagglers to work across a diverse range of conversations. The main purpose of this final project is to understand the basics of deep learning techniques applied to NLP. So it would be more doable to work on a project in such a limited time for which there exist many established references/documents. 

 
### Description of The Competition
Taking advantage of Kaggle's TPU support, this competition aims to build multilingual models with English-only training data. The model will be tested on Wikipedia talk page comments in several different languages. It is supported by [The Conversation AI team], funded by [Jiasaw](https://jigsaw.google.com/) and Google. Its mission is *to create future-defining research and technology to keep our world safer*. 

### Evaluation Metrics and Submission Requirements
Basically it is a classification problem. The model performance is evaluated by the [area under the ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predictions and the observations.

The submission file consists of two columns. The first column indicates the comment `id` and the second one is the probability for the `toxicity` variable. Following is an example submission file.
<table style="width:100%">
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


## <a href="#part-2-eda" name="part-2-eda">Part 2: EDA </a>

### Dataset

### Preprocessing

### EDA

## <a href="#part-3-basics-of-language-models" name="part-3-basics-of-language-models">Part 3: Basics of Language Models </a>

### What is a Language Model?

### Word Embeddings

### Self Attention

<!-- ### BERT -->
