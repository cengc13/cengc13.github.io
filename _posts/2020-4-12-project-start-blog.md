---
layout: post
title: Jiasaw Multilingual Toxic Comment Classification: Starter Blog
---


This project is also an ongoing Kaggle Competition, whihc lives at [here](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification). It is a natural language processing (NLP) task. I chose this topic as the final project because NLP is a very hot topic nowadays and I am new to this area. I hope to take advantages of this opportunity to learn more about deep learning targeted towards the state-of-art application in NLP. 

<div style="font-size:75%; background-color:#eee; border: 1px solid #bbb; display: table; padding: 7px" markdown="1">

<div style="text-align:center" markdown="1">  

**Contents**

In this starter blog, I will walk you through the background, evaluation metrics, submission requirement, exploratory data analysis, and possible state-of-art language models for this project. The overview will be as follows.

</div>

* **[Part 1: Introduction](#introduction-and-background)**
  * What is a Language Model
  * Transformers for Language Modeling
  * One Difference From BERT
  * The Evolution of The Transformer Block
  * Crash Course in Brain Surgery: Looking Inside GPT-2
  * A Deeper Look Inside
  * End of part #1: The GPT-2, Ladies and Gentlemen
* **[Part 2: EDA](#part-2-illustrated-self-attention)**
  * Self-Attention (without masking)
  * 1- Create Query, Key, and Value Vectors
  * 2- Score
  * 3- Sum
  * The Illustrated Masked Self-Attention
  * GPT-2 Masked Self-Attention
  * Beyond Language modeling
  * You've Made it!
* **[Part 3: Language models](#part-3-beyond-language-modeling)**
  * Machine Translation
  * Summarization
  * Transfer Learning
  * Music Generation

</div>

## Part 1: Introduction <a href="#introduction and background" name="part1-introduction-and-background">#</a>