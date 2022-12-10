![](UTA-DataScience-Logo.png)

# Drone Delivery 

* This repository holds an attempt to apply model-free Reinforcement Learning techniques to optimize product delivery times as detailed in the "Drone Delivery" Kaggle challenge (https://www.kaggle.com/competitions/hashcode-drone-delivery/overview). 

## Overview

  * **Definition of the tasks / challenge**  Ex: The task, as defined by the Kaggle challenge is to use a time series of 12 features, sampled daily for 1 month, to predict the next day's price of a stock.
  * **Drone Delivery challenge**  The task, as defined by the Kaggle challenge, is to develop code that will read a variable input file of system setup parameters and customer orders, and subsequently generate a submission file detailing each action that will be performed. The submission file will subsequently be scored by Kaggle based on how rapidly each order was delivered.   
  * **Solution method** The approach in this repository leverages reinforcement learning (RL) to determine an efficient path to complete the orders. The problem was identified as fitting an infinite-horizon, discounted return equation which aligns well with model-free deep RL algorithms.  I compared the performance of 2 different RL algorithms, PPO and A2C.
  * **Summary of the performance achieved** Ex: Our best model was able to predict the next day stock price within 23%, 90% of the time. At the time of writing, the best performance on Kaggle of this metric is 18%.

## Summary of Workdone

Include only the sections that are relevant an appropriate.

### Data

* Data:
  * Type: For example
    * Input: CSV configuration file containing challenge metrics (all data are integers):
> * Size of delivery area
> * Number of drones
> * Number of warehouses
> * Number of product types and weight for each type
> * Inventory of each warehouse
> * Order list detailing customer location, product numbers and quantities
 
  * Size: Training set consists of 30 drones, 10 warehouses, 400 product types, 1250 orders for 9396 total 
items.
  * This Kaggle challenge requires a code (notebook) submission which will process the test data set for subsequent scoring. 

#### Preprocessing / Clean up

* No cleaning of the data was necessary for the training data set. All values were positive (or zero) integers. All orders were for corresponding valid product numbers. 

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







