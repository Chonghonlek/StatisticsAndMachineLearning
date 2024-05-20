# StatisticalLearning

Here we apply techniques from 
1) **Gareth J et al., An introduction to Statistical Learning**
2) 


and implement them in python

## What are we trying to solve in Stats learning

There is some relation between Response $Y$ and predictors $ X = (X_1 \cdots X_P)$ in the form $Y = f(X) + \epsilon$

Here $f$ is some fixed but unknown function of $X$ and $\epsilon$ is the error term indepent of $X$ and has mean = 0. $f$ represents the systematic information that $X$ provides about $Y$.

Statistical learning refers to the approaches for estimating $f$

## Why estimate $f$

1) Prediction $\hat{Y} = \hat{f}(X)$

Accuracy of $\hat{Y}$ depends on reducible and irreducible error. $\hat{f}$ is generally not a perfect estimate for $f$, this error is reducible as we can improve the accuracy of $\hat{f}$

Irreducible error stems from the variability of $\epsilon$

$$
\begin{aligned}
E(Y - \hat{Y})^2 &= E(f(X) + \epsilon - \hat{f}(X))^2 \\
&= \underbrace{[f(X) - \hat{f}(X)]^2}_{\text{reducible error}} +\underbrace{Var(\epsilon)}_{\text{irreducible error}} \quad \because Var(\epsilon) = E(\epsilon^2) - E(\epsilon)^2 = E(\epsilon^2)
\end{aligned}
$$

2) Inference

    * (E.g.) what predictors are associated with the response - ie finding the few important predictors $X_k$  among all possible variavbes $X_1 \cdots X_n$
    * (E.g.) What is the r/s between response and each predictor - ie the correlation between variables X and Y
    * (E.g.) Can r/s between $Y$ and each predictor be adequatly summarised using a linear equation. Or is the relationship more complicated? - ie what is the complexity of model


## Estimating $\hat{f}$
Most statistical learning methods can be characterised as either parametric or non-parametric

In parametric approach, we make assumption about the function form or shape of $f$ and we estimate its parameters by fitting training data. This approach redices the problem of estimating $f$ down to estimating a set of praremeters. The disadvantage of a parametric approach is that the model we choose will usually not match the true unknown form of $f$. We can address this problem with more flexible models but this will require estimating more parameters and lead to overfitting(following noise too closely)

In the non-parametric approach, we dont make assumptions about the form of $f$. Instead we seek an estimate of f that gets close to the data points as possible. By avoiding the assumption of a particular form for $f$, they have the potential to accruately fit a wider range fo psossible shapes for $f$. But the non-parametric approach may suffer from a disadvatange - which is a large number of observations will be required in order to obtain an accurate estimation of $f$. This approach is also prone to overfitting

### Trade offs in prediction accuracy and model interpretability

We may prefer a more restrictive method instead of a very flexible approach (eg more parameters) due
to the 
* higher interpretability of a restrictive model. want to understand how any individual predictior is associated with the response
* reduction in chances of overfitting


### Unsupervised vs Supervised learning

In supervised learning we require a label, where we fit a model that relates the response to the predictors. Used to solve the issue of classification

In contrast, unsupervised learning we have no respponse variable to predicd. used to solve the issue of clustering - eg market/consumer segmentation

Many problems fall naturally into supervised or unsupervised learning paradigms

### Regression vs Classification

Problems with quantitative response are typically refered as regression problems while those involving a qualitative response are often referred as classification problems.

We tend to select statistical learning methods on the basis of whether the repsonse is quantitative or qualitative. Whether the predictors are quantitative or qualitative is consiered less important


## Assesing model accurarcy

Selecting the best model is one of the most challenging parts of performing statistical learning

### Measuring quality of fit.  

We need to quantify the extent to which the predicted response for an observation is close to the true response. In regression we use, the _mean squared error_ (MSE) :

$$
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{f}(x_i))^2
$$

The MSE computed using the training data so more accurately be referred to as the _training MSE_. We want to know, more importantly for prediciton, whether $\hat{f}(x_0)$ is approximately equal to $y_0$ where $(x_0,y_0)$ is a previously unseen test observation. We aim to choose the method that gives the lowest _test MSE_ as opposed to the lowest _training MSE_

In other words we compute the averaged squared prediction error : $Avg(y_0 - \hat{f}(x_0))^2$ for the test observations. Select the model that minimises this error.

If no test data is available, we select the learning method that minimises the _training MSE_ instead

![Taken from Gareth J et al](images/Screenshot%202024-05-20%20Figure2.9%20G%20James.png)

From the above figure, we see the tradeoff between flexibility and MSE. As we add in more parameters/Degree of freedoms, training MSE falls. However, test MSE will only fall until a certain point, after which test MSE rises due to overfitting of the model. ie when a small trainig MSE leads to a large test MSE we say this to be overfitting the data. 

Regardless of whether overfitting has occured, we would expect the training MSE to be lower than the test MSE, since the statistical learning methods seeks to minimise the training MSE in the first place. 

This is a fundamental property in statistical learning.

In practice, one usually compute the taining MSE with ease, but estimating test MSE is more difficult because usually no test data are availbale. One important method is cross validation - a method to estimate test MSE using training data.

### Bias-Variance trade off

The U shape observed in the testMSE turns out to be the result of 2 competing properties of statisical learning methods. We can decompose _Expected test MSE_ or _average test MSE_ into the form :

$$
E(y_0 - \hat{f}(x_0))^2  = Var(\hat{f}(x_0)) + [Bias(\hat{f}(x_0))]^2 + Var(\epsilon)
$$

<details>
<summary>Note</summary>

$$
\begin{aligned}
\text{Bias}(\hat{\theta}) &= E(\hat{\theta}) - \theta \\
\\
E[(\hat{y} - y)^2] &= E[(\hat{y} - E[\hat{y}])^2] + (E[\hat{y}] - y)^2 \\
MSE & = Variance + Bias^2               
\end{aligned}
$$

Possible heuristic:

MSE = how far model is from actual 

Variance = how far model is from average model

$Bias^2$ = how far is the average model from the actual

Derivation:

$$
\begin{aligned}
E[(y - \hat{y})^2] &= E[(f(x) - \hat{f}(x))^2] + \sigma_\epsilon^2 \\
\\
E[(f(x) - \hat{f}(x))^2] &= E[((f(x)-E[\hat{f}(x)])- (\hat{f}(x)- E[\hat{f}(x)] ))^2]  \\
&= E[(E[\hat{f}(x)]- f(x))^2] + E[(\hat{f}(x)- E[\hat{f}(x)] )^2] \\
&-2E[(f(x)-E[\hat{f}(x)])(\hat{f}(x)- E[\hat{f}(x)])]\\
&= \underbrace{(E[\hat{f}(x)]- f(x))^2}_{bias^2} + \underbrace{E[(\hat{f}(x)- E[\hat{f}(x)] )^2]}_{variance} \\
&-2(f(x)-E[\hat{f}(x)])(E[\hat{f}(x)] - E[\hat{f}(x)]) \\
\because E[(E[\hat{f}(x)]- f(x))^2] \quad &\text{is the expectation of a constant}\\
=& \text{Bias}[\hat{f}(x)]^2 + Var(\hat{f}(x))
\end{aligned}
$$


</details> <br>

The equation tells us that in order to minimise the expected squared error, we need to have low variance and low bias. 

Variance here refers to the amount by which $\hat{f}$ will change due if we estimated it using a different trainig data set. high variance imply small change in data lead to large change in $f$. Model with high variance is highly flexible and capture patterns but it can also capture noise - lead to poor genealization performance

Bias refers to the error introduced by approximating a real life problem. Eg linear regression has a high bias given its simplicity. High bias methods makes strong assumptions about underlying data, they tend to be simple and may not capture complexities in data. On the other hand, more flexible methods result in lower bias. 

As we use more flexible methods, variance increases and biases decreases. The relative rate of change determine whether the test MSE rise or falls. At some point, increasing flexibility has little impact on the bias but starts to increase the variance, causing test MSE to rise. 

Good test set performance requried both low variance as well as low squared bias. The challenge lies in finding a method that for which both the variance and low squared bias are low.

In real life in which f is unobserved, it is generally not possible to explicityly compute the test MSE,Bias,Variance for a method.

### Accuracy with the classification setting
Suppose now we have $y_i$ that is no longer quantitative but qualitative. The common approach for quantifying the accuracy of our estimate $\hat{f}$ is the _training error rate_ or the proportion of mistakes that are made if we apply our estimate $\hat{f}$ to the training observation:

$$
\frac{1}{n} \sum_{i=1}^n I(y_i \neq \hat{y_i})
$$

Here $\hat{y_i}$ is the predicted class label from $\hat{f}$. The above computes the fraction of incorrect classifications

The _test error_ associated with a test of observations of the form $(x_0,y_0)$ is given by $Avg(I(y_i \neq \hat{y_i}))$. A good classifier is one that has the smallest _test error_

#### Bayes classifier

We can show that the above test error can be minimised on average by a classifier that assigns each observation to the most likely class given its predictor values. We assign a test observation with predictor vector $x_0$ to the class j for which $P(Y=j|X=x_0)$ is the largest. In a simple 2 class scenario, this classifier corresponds to predicting class 1 if $P(Y=1|X=x_0) > 0.5$ and class 2 otherwise

The Bayes' classifier's prediciton is determined by the bayes decision boundary. The Bayes classifier produces the lowest possible test error rate called the Bayes error rate. Since the Bayes classifier always chooses the class for which the probability is the larget, the error rate will be $1 - max_j P(Y=j|X=x_0)$. In general the overall Bayes error rate is given by :

$$
1 - E[max_j P(Y=j|X)] , \text{where the expectation averages the probability over all X}
$$

#### K-Nearest Neighbour - KNN

For real data, we do not know the conditional distribution of Y given X and so computing the Bayes classififier is impossible, Therefore, the bayes classifier serves as an unattainable gold standard.

One way to estimate the conditional distribution of Y given X and classify a given observation to the class with highest estimated probability is the **KNN classifier**

KNN Classifier first identifies the K (+ve int) points in the training data closest to $x_0$ given by $\Nu_0$. It then estimates the conditional probability for class j as a fraction of points in $\Nu_0$ whose response values equal j:

$$
P(Y = j|X= x_0) = \frac{1}{K} \sum_{i \in \Nu_0} I(y_i = j)
$$

KNN then classifies the test observation $x_0$ to the class with the class with the largest probability above.

Suppose K = 3, KNN identifies the 3 observations closest to the point x_0 and computes the estimated probability of x_0 being in one of the classes based on the 3 closest observations and their class.

In fact, we can see that the KNN decision boundary is similar to the bayes decision boundary.

![KNN](images/Screenshot%202024-05-20%20Figure2.16%20Gareet%20J.png)

However when K = 1, we see that the decision boundary is overly flexible and finds patterns that dont correspond to the bayes decision boundary, this corresponds to low bias and high variance.
As K increases, the method becomes less flexible and produices a decision boundary that is close to linear. This corresponds to high bias and low variance.

Plotting the KNN training error and test error against level of flexibility (1/K on the log scale) shows that training error continues to decline with increasing flexibility ie K decreasing and a U-shape for the test error. 

in both regression and classification, choose the level of flexibility is critical to success of any statistical learning method. There are ways to estimate test error rates and choosing the level of flexibility for a given method (to be discussed later).




