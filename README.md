# Understanding Decision Tree Regression in Machine Learning

Decision Tree Regression is a powerful machine learning algorithm used for predicting continuous values. Unlike its counterpart, Decision Tree Classification, which is designed for discrete outcomes, Decision Tree Regression is tailored to handle regression problems. In this tutorial, we will explore the fundamentals of Decision Tree Regression, its structure, and how to implement it using Python's scikit-learn library.

## 1. Introduction to Decision Tree Regression:
Decision Tree Regression is a supervised learning algorithm that is capable of handling both numerical and categorical data. It works by recursively partitioning the dataset into subsets based on the values of input features. The ultimate goal is to create a predictive model represented as a tree structure.
YouTube: https://youtu.be/-sY5jlQTbek?si=bzN_wKpWqU8zoLur

## 2. How Decision Tree Regression Works:
The algorithm works by recursively splitting the dataset based on the values of features, selecting the feature that provides the best split at each node. Each internal node of the tree represents a decision based on a specific feature, and each leaf node contains the predicted continuous value.

## 3. Key Components of a Decision Tree:
Root Node: The topmost node in the tree.
Internal Nodes: Nodes that represent decisions based on features.
Leaf Nodes: Terminal nodes that contain the predicted values.
Edges/Branches: Connections between nodes representing decisions.

## 4. Splitting Criteria:
Decision Tree Regression uses splitting criteria such as Mean Squared Error (MSE) to determine the best feature and value to split the dataset at each node. MSE measures the variance of the target variable within each subset.

# Decision Tree Regression Example using MSE

Let's walk through a simple example of Decision Tree Regression using Mean Squared Error (MSE) as the splitting criterion.

Consider a dataset with a single feature, X, and a target variable, y:

| X | y |
|---|---|
| 2 | 10 |
| 4 | 15 |
| 5 | 12 |
| 8 | 20 |

Now, let's build a decision tree.

## Step 1: Calculate Initial MSE

Calculate the initial MSE for the entire dataset. The formula for MSE is:

\[ MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2 \]

where \( N \) is the number of samples, \( y_i \) is the actual target value for each sample, and \( \bar{y} \) is the mean target value.

For the initial dataset:

\[ \bar{y} = \frac{10 + 15 + 12 + 20}{4} = \frac{57}{4} \]

\[ MSE = \frac{1}{4} [(10 - \frac{57}{4})^2 + (15 - \frac{57}{4})^2 + (12 - \frac{57}{4})^2 + (20 - \frac{57}{4})^2] \]

\[ MSE = \frac{1}{4} [(\frac{23}{4})^2 + (\frac{3}{4})^2 + (\frac{-9}{4})^2 + (\frac{23}{4})^2] \]

\[ MSE = \frac{1}{4} [\frac{529}{16} + \frac{9}{16} + \frac{81}{16} + \frac{529}{16}] \]

\[ MSE = \frac{1}{4} \times \frac{1148}{16} = \frac{287}{4} \]

## Step 2: Evaluate Potential Splits

Consider each possible split point for the feature X and calculate the MSE for each split.

### Split at X = 3:

Subset 1: {(2,10), (4,15)}  
Subset 2: {(5,12), (8,20)}

\[ MSE_{\text{left}} = \frac{1}{2} [(10 - \frac{25}{2})^2 + (15 - \frac{25}{2})^2] \]

\[ MSE_{\text{right}} = \frac{1}{2} [(12 - \frac{32}{2})^2 + (20 - \frac{32}{2})^2] \]

\[ Total \, MSE = MSE_{\text{left}} + MSE_{\text{right}} = \frac{1}{2} [\frac{25}{4} + \frac{25}{4}] + \frac{1}{2} [\frac{64}{4} + \frac{64}{4}] \]

\[ Total \, MSE = \frac{1}{2} [\frac{50}{4}] + \frac{1}{2} [32] \]

\[ Total \, MSE = \frac{1}{2} [\frac{25}{2}] + \frac{1}{2} [32] \]

\[ Total \, MSE = \frac{25}{4} + 16 = \frac{81}{4} \]

### Split at X = 4:

Subset 1: {(2,10)}  
Subset 2: {(4,15), (5,12), (8,20)}

\[ MSE_{\text{left}} = 0 \]

\[ MSE_{\text{right}} = \frac{1}{3} [(15 - \frac{47}{3})^2 + (12 - \frac{47}{3})^2 + (20 - \frac{47}{3})^2] \]

\[ Total \, MSE = MSE_{\text{left}} + MSE_{\text{right}} = 0 + \frac{1}{3} [\frac{529}{9} + \frac{121}{9} + \frac{529}{9}] = \frac{25}{3} \]

### Split at X = 5:

Subset 1: {(2,10), (4,15)}  
Subset 2: {(5,12), (8,20)}

\[ MSE_{\text{left}} = \frac{1}{2} [(10 - \frac{25}{2})^2 + (15 - \frac{25}{2})^2] \]

\[ MSE_{\text{right}} = \frac{1}{2} [(12 - \frac{28}{2})^2 + (20 - \frac{28}{2})^2] \]

\[ Total \, MSE = MSE_{\text{left}} + MSE






