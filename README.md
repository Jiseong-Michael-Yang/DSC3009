# Predicting Bank Marketing Outcome
This is a documenation about deep learning project to predict if a customer would subsribe a term deposit of a bank or not, with Tensorflow.

## 1. Project Overview
* Project Objective
   * Prediction based on the classification with logistic regression
   * Target Feature
      * y: has the client subscribed a term deposit? 
      * value: binary(yes or no)

* About the Dataset
    * Dataset from UCI Machinie Learning Repository.
        * https://archive.ics.uci.edu/ml/datasets/bank+marketing
    * Customer data from May 2008 to November 2010.
    * 41188 rows with 20 columns.

* About the features
    * Please refer to the attribute information [here](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
    
## 2. Data Pre-processing

 * The target feature was encoded {yes: 1, no: 0}.
 * Categorical varibles were all transformed into dummy variables.
 * Insignificant features detected were removed as per the model summary result on logidtic regression by R. 
 * Insignificant featuers again removed as per the result of correlation plot by R. 
 * All numerical features standardized by Standard Scaler by 'sklearn'.
 * Training set and test set of 8:2 ratio.
    
## 3. Modeling

![image](https://user-images.githubusercontent.com/46237445/50608069-b1e45200-0f0e-11e9-8294-d8716e43876f.png)

  * 9 input variables
  * K input between hidden layers
  * 1 output variable
  * Xavier Initializer
  * Leaky ReLU/ReLU & Sigmoid
  * AdamOptimizer
  * Cost function for logistic regression

## 4. Model Optimization

  * Hyper Parameters
  
    Hyper Parameters | Value |
    :--------------: | :---: |
    Learning Rate | 0.003 |
    Dropout Rate | 0.5 ~ 0.7 |
    Threshold | 0.65 |
    Number of Layers | 5 ~ 7|
    Number of Inputs | 27 ~ 45 |
    Iterations | 1000 |
    
  * Finding the Optimal Cutoff Value (ROC Curve)
    * Optimal cutoff value of 0.65
  
  ![image](https://user-images.githubusercontent.com/46237445/50656796-af016400-0fd7-11e9-9854-6fbb7c7d545c.png)
   
  * Hyperparameter Tunning
    * The accuracy ranges from 0.88 ~ 0.91
    * The highest accuarcy was 0.9114 with 6 layers, 45 inputs and the dropout rate of 0.6.
  
  ![image](https://user-images.githubusercontent.com/46237445/50657041-92196080-0fd8-11e9-8002-c21bcdc5fe40.png)
  
## 5. Conclusion

  * The Sinnificance of the Project 
    * Bank profit = loan interest - deposit interest
    * Optimize the profit by offering customized financial product
    
  * Further Improvements
    * Batch traning would allows us to efficiently reduce the cost value and the accuracy
    * Visualization of the training process with Tensorboard
