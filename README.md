# Deep Learning Specialization
Instructor of the specialization: [Andrew Ng](http://www.andrewng.org/)

### Table of Contents
1. [My Learnings from the Specialization](#Learning)
2. [Instructions to use the repository](#Instruction)
3. [Weekly Learning Objective](#Description)
4. [Results](#res)
5. [Disclaimer](#Disc)

## My Learnings from the Specialization<a name="Learning"></a>

In this five course series, I learned about the foundations of `Deep Learning` by implementing vectorized **neural networks (MLP, CNN, RNN, LSTM) and optimization algorithms (SGD, RMSprop, Adam)** from scratch in Python, building and training deep neural networks in **TensorFlow and Keras** and identifying key parameters in network architecture for hyperparameter tuning.

I learned about the best practices to train and develop test sets and analyzed `bias/variance` for building DL applications, diagnosed and used strategies for reducing errors in ML systems, understand complex ML settings and used **transfer learning for image classification tasks**.

I learned to build and train CNN models **(YOLO for object detection, U-Net for image segmentation, FaceNet for face verification and face recognition)** for visual detection and recognition tasks and to generate art work through neural style transfer by using a pre-trained VGG-19 model. I learned about RNNs, GRUs, LSTMs and transformers and applied them to various NLP/sequence tasks. I used RNNs to built a character-level language model to generate dinosaur names, **LSTMs to built a Seq2seq model for Neural Machine Translation with attention and trigger word detection model**. I used pre-trained transformer models for question-answering and named-entity-recognition tasks.

## Instructions to use the repository<a name="Instruction"></a>
Using this repository is straight forward. Clone this repository to use.
This repository contains all my work for this specialization. All the code base, quiz questions, screenshot, and images, are taken from, unless specified, [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning?utm_source=gg&utm_medium=sem&utm_campaign=17-DeepLearning-US&utm_content=17-DeepLearning-US&campaignid=904733485&adgroupid=49070439496&device=c&keyword=neural%20network%20for%20machine%20learning&matchtype=b&network=g&devicemodel=&adpostion=&creativeid=415429113789&hide_mobile_promo&gclid=EAIaIQobChMI5_CtgI_t7wIVPObjBx0xuwp6EAAYASAAEgKLhvD_BwE).

## Weekly Learning Objective<a name="Description"></a>
1. **[Course 1 - Neural Networks and Deep Learning](https://github.com/Ankit-Kumar-Saini/Deep_Learning_Specialization/tree/main/C1%20-%20Neural%20Networks%20and%20Deep%20Learning)**

- **Course Objective:** This course focuses on vectorized implementation of neural networks in Python.

   - **Week 1: Introduction to deep learning**
      - Be able to explain the major trends driving the rise of deep learning, and understand where and how it is applied today.

   - **Week 2: Neural Networks Basics**
      - Python Basics with Numpy and Logistic Regression with a Neural Network mindset.

   - **Week 3: Shallow neural networks**
      - Understand the key parameters in a neural network's architecture. Planar data classification with a hidden layer

   - **Week 4: Deep Neural Networks**
      - Understand the key computations underlying deep learning, use them to build and train deep neural networks, and apply it to computer vision.

2. **[Course 2 - Improving Deep Neural Networks Hyperparameter tuning, Regularization and Optimization](https://github.com/Ankit-Kumar-Saini/Deep_Learning_Specialization/tree/main/C2%20-%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization)**

- **Course Objective:** This course teaches the "magic" of getting deep learning to work well. Rather than the deep learning process being a black box, you will understand what drives performance, and be able to more systematically get good results. 

   - **Week 1: Practical aspects of Deep Learning**
      - Understand industry best-practices for building deep learning applications. Be able to effectively use the common neural network "tricks", including initialization, L2 and dropout regularization, Batch normalization, gradient checking along with implementation.

   - **Week 2: Optimization algorithms**
      - Be able to implement and apply a variety of optimization algorithms, such as mini-batch gradient descent, Momentum, RMSprop and Adam, and check for their convergence. 

   - **Week 3: Hyperparameter tuning, Batch Normalization and Programming Frameworks**
      - Understand new best-practices for the deep learning era of how to set up train/dev/test sets and analyze bias/variance. 
      - Implement a neural network in TensorFlow.

3. **[Course 3 - Structuring Machine Learning Projects](https://github.com/Ankit-Kumar-Saini/Deep_Learning_Specialization/tree/main/C3%20-%20Structuring%20Machine%20Learning%20Projects)**

- **Course Objective:** This course focuses on how to diagnose errors in a machine learning system, be able to prioritize the most promising directions for reducing error, understand complex ML settings, such as mismatched training/test sets and comparing to and/or surpassing human-level performance and how to apply end-to-end learning, transfer learning, and multi-task learning. 

   - There is no Programming Assignment for this course. But this course comes with very interesting case study quizzes.

4. **[Course 4 - Convolutional Neural Networks](https://github.com/Ankit-Kumar-Saini/Deep_Learning_Specialization/tree/main/C4%20-%20Convolutional%20Neural%20Networks)**

- **Course Objective:** This course focuses on how to build a convolutional neural network, including recent variations such as residual networks, how to apply convolutional networks to visual detection and recognition tasks and use neural style transfer to generate art.

   - **Week 1 - Foundations of Convolutional Neural Networks**
      - Build Convolutional Model in python from scratch.

   - **Week 2 - Deep convolutional models: case studies**
      - Build Residual Network in Keras.
      - Transfer Learning with MobileNet.

   - **Week 3 - Object detection**
      - Learn how to apply your knowledge of CNNs to one of the toughest but hottest field of computer vision: Object detection. Autonomous driving application - Car detection.
      - Image segmentation with U-Net.

   - **Week 4 - Special applications: Face recognition & Neural style transfer**
      - Discover how CNNs can be applied to multiple fields, including art generation and face recognition. 
      - Build Face Recognition model for the Happy House. Implement Art Generation with Neural Style Transfer.

5. **[Course 5 - Sequence Models](https://github.com/Ankit-Kumar-Saini/Deep_Learning_Specialization/tree/main/C5%20-%20Sequence%20Models)**

- **Course Objective:** This course focuses on how to build and train Recurrent Neural Networks (RNNs), and commonly-used variants such as GRUs and LSTMs, able to apply sequence models to natural language problems, including text synthesis and sequence models to audio applications, including speech recognition and music synthesis.

   - **Week 1 - Recurrent Neural Networks**
      - Build a Recurrent Neural Network in python from scratch. Implement Character-Level Language Modeling to generate Dinosaur names. Generate music to Improvise a Jazz Solo with an LSTM Network.

   - **Week 2 - Natural Language Processing & Word Embeddings**
      - Using word vector representations and embedding layers you can train recurrent neural networks with outstanding performances in a wide variety of industries. Examples of applications are sentiment analysis, named entity recognition and machine translation.

   - **Week 3 - Sequence models & Attention mechanism**
      - Sequence models can be augmented using an attention mechanism. This algorithm will help your model understand where it should focus its attention given a sequence of inputs.
      - Implement Neural machine translation with attention and Trigger word detection.

  - **Week 4 - Transformer Network**
      - Use HuggingFace tokenizers and transformers to perform Named Entity Recognition and Question Answering


## Results<a name="res"></a>
`Some results from the programming assignments of this specialization`

- Image classification using Logistic Regression from scratch in Python
![alt text](https://github.com/Ankit-Kumar-Saini/Coursera_Deep_Learning_Specialization/blob/main/Results/logistic_reg.PNG) 

- Accuracy vs number of hidden layers in MLP for planar data set
![alt text](https://github.com/Ankit-Kumar-Saini/Coursera_Deep_Learning_Specialization/blob/main/Results/hidden_layers.PNG) 

![alt text](https://github.com/Ankit-Kumar-Saini/Coursera_Deep_Learning_Specialization/blob/main/Results/hidden_layer_2.PNG) 


- YOLO object detection
![alt text](https://github.com/Ankit-Kumar-Saini/Coursera_Deep_Learning_Specialization/blob/main/Results/object_detection.PNG) 

- Neural style transfer
![alt text](https://github.com/Ankit-Kumar-Saini/Coursera_Deep_Learning_Specialization/blob/main/Results/neural_style_transfer.PNG)

## Disclaimer<a name="Disc"></a>
The solutions uploaded in this repository are only for reference when you got stuck somewhere. Please don't use these solutions to pass programming assignments. 


