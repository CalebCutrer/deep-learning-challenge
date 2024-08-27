# deep-learning-challenge
Module 21 Challenge

# Alphabet Soup Deep Learning Model Performance Report
## Overview
The objective of this project was to develop a machine learning model that could predict the success of funding applications for Alphabet Soup. Leveraging a dataset containing historical records of previously funded organizations, I constructed a binary classification model to forecast the likelihood of future funding success based on key features provided in the data. The focus was on utilizing deep learning techniques, specifically a neural network, to create this predictive model and fine-tune its performance to reach an accuracy of 75% or higher.

## Results 
### Data Preprocessing
* Target Variable: The target variable for this model was IS_SUCCESSFUL, which indicates whether an organization was successful after receiving funding.
* Feature Variables: the features used in the model are listed below
    * APPLICATION_TYPE
    * AFFILIATION
    * CLASSIFICATION
    * USE_CASE
    * ORGANIZATION
    * STATUS
    * INCOME_AMT
    * SPECIAL_CONSIDERATIONS
    * ASK_AMT
* Removed Variables: As per the instructions, the EIN and NAME columns were removed from the dataset because they are ID numbers and wouldn't contribute to the model.

## Compiling, Training, and Evaluating the Initial Model and the Subsequent Optimization Iterations  
### Initial Model Structure:
* First hidden layer: 80 neurons, relu activation
* Second hidden layer: 30 neurons, relu activation
* Output layer: 1 neuron, sigmoid activation

Training and Performance Results: Trained with 100 epochs, the original model achieved an accuracy of 72.49% with a loss of 0.5673.

### First Optimization Model Structure:
* First hidden layer: 120 neurons, relu activation
* Second hidden layer: 60 neurons, relu activation
* Third hidden layer: 60 neurons, relu activation
* Output layer: 1 neuron, sigmoid activation

Training and Performance Results: Trained with 100 epochs, the first optimization model achieved an accuracy of 72.75% with a loss of 0.5845. Thus, the added hidden layer and increases in neurons provided a marginal increase in accuracy over the initial model. 

### Second Optimization Model Structure:
* First hidden layer: 120 neurons, silu activation
* Second hidden layer: 60 neurons, silu activation
* Third hidden layer: 10 neurons, silu activation
* Fourth hidden layer: 60 neurons, silu activation
* Output layer: 1 neuron, sigmoid activation

Training and Performance Results: Trained with 120 epochs, the second optimization model achieved an accuracy of 72.55% with a loss of 0.5895. Thus, increrasing the number of epochs, changing the activation type, adding another hidden layer and increases in layer neurons provided no discernable increase in accuracy over the previous two models. 

### Third Optimization Model Structure:
* First hidden layer: 120 neurons, linear activation
* Second hidden layer: 90 neurons, linear activation
* Third hidden layer: 10 neurons, relu activation
* Fourth hidden layer: 60 neurons, relu activation
* Output layer: 1 neuron, sigmoid activation

Training and Performance Results: Trained with 150 epochs, the third optimization model achieved an accuracy of 72.94% with a loss of 0.5587. Thus, increrasing the number of epochs again, changing the activation type for several of the hidden layers, and further increasing the number of neurons in the hiddden layers provided a very slight increase in accuracy over the previous models. 


## Summary
The final deep learning model achieved an accuracy of about 73%. While this is a minor improvement over the initial model, the final model was still insufficient at meeting the target performance of 75% accuracy. Thus. there was marginal (if any) progress made in the several attempts at optimizing the network structure through adding epochs, changing activation types, adding hidden layers, and increasing the neurons in said layers; the model still fell short of meeting the 75% accuracy threshold.
Given the model modifications were unable to provide any meaningful improvement to the accuracy, it seems that it would be beneficial to try alternative machine learning models such as Support Vector Machines (SVM). I suggest this because it seems as though this type of model has a ceiling around 73% accuracy, it may be limited to that level of accuracy. 