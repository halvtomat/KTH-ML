# DD2421 Machine Learning Exam

## Student

Daniel Gustafsson

danielg8@kth.se

199808275110

## A Graded problems

### A-1 Terminology

**a)** **2** RANSAC == Robust method to fit a model to data with outliers

**b)** **10** *k*-means == Clustering method based on centroids

**c)** **1** *k*-fold cross validation == A technique for assessing a model while exploiting available data for training and testing

**d)** **9** Fisher's criterion == An approach to find useful dimension for classification

**e)** **3** Curse of dimensionality == Issues in data sparsity in space

### A-2 Nearest Neighbor, Classification

**a)** 15%

**b)** 16%, Since the training set in 1-nearest neighbor **is** the model and each point is in its own nearest neighbor-zone there can't be any training error.

**c)** I would use the decision forest classifier because of the lower test error rate.

**d)** No the reasoning would most likely **not** hold since now there could potentially be training errors because points of class A could lie in a zone where there is a majority of class B points.

### A-3 Regression with regularization (LASSO)

**a)** **iii**, When lambda is extremely big, the model will just be a constant function with 0 variance. When lambda is 0 the model will just be regular linear regression which has high variance.

**b)** Variable selection property, LASSO can zero out some properties making the model more sparse.

### A-4 PCA, Subspace Methods

**a)** Largest eigenvalues

**b)** 

**c)** Gamma

### A-5 Information gain

**a)** **60%**, 50% + 20% - 10% (p(A or B) = p(A) + p(B) - p(A and B))

**b)** **0.971**, -(0.4\*log2(0.4) + 0.6\*log2(0.6))

**c)** **0.249**, entropy before checking website = 0.971, entropy after checking website = 0.722, information gain = 0.971 - 0.721 = 0.249

## B Graded problems

### B-1 Warm up

Shoogee has a 9% probability to hide in any of the hiding spots so now that 8 of them are removed from the total probability, the remaining probability that she is in one of those spots is 9% * 2 = 18%. Additionally the probability that she is in a new spot was initially 10% so the total remaining probability is 28%. 

**Probability that she has found a new hiding spot is 0.1 / 0.28 = 0.36 = 36%**

### B-2 Maximum likelihood estimation

### B-3 Probabilistic classification

## C Graded problems

### C-1 Support Vector Machine

**a)** The support vectors will be the following three points: (-1, -1), (1, -1), (0, 1). The margin will be between these points.

**b)**

**c)** A non-linear kernel could provide a wider decision boundary margin which is better because it produces more stable results and has less variance.

### C-2 Artificial Neural Networks