Notes taken from  **Gareth J et al., An introduction to Statistical Learning**

Resampling involves drawing samples from a training set and refitting a model of interest on each sample to obtain additional information about the fitted model. 

We discuss 2 commonly used resampling methods: cross-validation and bootstrpa.

- cross validation can be used to estimate the test error associated with a given statistical learning method to evaluate its performance(model assessment) or to select the apporpriate level of flexibility(model selection)
- bootstrap is used in several contexts, most commonly to provide a measure of accuracy of a parameter estimate of a given statistical learning model

# Cross Validation

The test error can be easily calculated if a designated test set is available. Unfortunately, this is usually not the case. 

In the absence of a very large designated test set that can be used to directly estimate the test error rate, a number of techniques can be used to estimate this quantity using the available training data. Some methods make a mathematical adjustment to the training error rate to estimate the test error rate (discussed in future chapters)

For this section, we consider a class of methods that estimate the test error rate by **holding out** a subset of the training observations from the fitting process, and then applying the statistical learning method to those held out observations

## Validation set approach

Here, we randomly divide the available set of observations into 2 parts - a training set and a validation/hold out set. The model is fitted on the training set and the fitted model is then used to predict responses in the validation set. 

The resulting validation set error rate , typically assessed using MSE - mean square error, in the case of quantitative response - provides an estimate of the test error rate. 

![fig 5.1 from G james et al](images/Screenshot%202024-07-21%20Figure5.1%20G%20James.png)

Above shows an example of how observations were split randomly into 2 equal sets. 

We can use the results to compare the impact/importance of regressors. eg if a cubic term will lead to lower MSE (better prediction) than using a quadratic term. We repeat using many different validation sets splits from the original observations to compare.
- but this will result in different estimations of the test MSE estimate. 

2 weaknesses for this approach:
- The validation estimate of the test error rate can be highly variable. it depends on precisely which observation are included in the training set and which are included in the validation set. (eg testing on influential data which not in training data)
- only a subset of observations - the training set are used to fit the model. Since statisical methods tend to perform worse when trained on fewer observations, this suggest that the validation set error may tend to **overestimate** the test error rate for the model fit on the entire data set

## Leave-One-Out Cross Validation (LOOCV)

The LOOCV approach attempts to address the drawbacks from the validation approach. 

Similarly, we need to split the set into 2 parts first. However, instead of creating 2 subset of comparable size, a single observation is used for validation, and the remaining observations make up the training set. 

The statisitical method if fitted on the n-1 training observations and a prediction $\hat{y_1}$ is made for the excluded observation. then we can get $MSE_1 = (y_1 - \hat{y_1})^2$ - that is approximately unbiased estimate for test error. but this is a poor estimate as it is highly variable - since it is based on a single observation

We can repeat this procedure by leaving out the other observation in each run. and train each run on the n-1 observations. Repeating this approach n times will give us: $MSE_1, MSE_2, \dotsc, MSE_n$ which we can then calculate the average of the n test error. The LOOCV estimate of test error is given by :

$$
CV_{n} = \frac{1}{n} \sum_i MSE_i
$$

![Fig 5.3 from G James et al](images/Screenshot%202024-07-21%20Figure5.3%20G%20James.png)

Advantages:
- it has far less bias. since we fitted almost the entire data set. instead of half in the validation approach. Thus LOOCV approach tends not to overestimate the test error rate as much as the validation set approach does. 
- In contrast to yielding different results when applied repeatedly due to randomness in training/validation splits, performing LOOCV multiple times always yield same results. (many rounds of LOOCV - mo randomness in dataset since no random split)

On the other hand, 
- LOOCV has potential to be expensive since the model has to be fitted n times. This can be very time consuming if n is very large and if the model is slow to fit. 

With **least squares linear or polynomial regression**, an amazing shortcut makes the cost of LOOCV the same as that as of a single model fit. 

$$
CV_{n} = \frac{1}{n} \sum_i (\frac{y_i - \hat{y_i}}{1- h_i})
$$

here $h_i$ refers to the leverage - which lies between 1/n and 1, it reflects the amount an observation influences its own fit. Hence, the residuals for high leverage points are inflated in this formula by exactly the right amount for this equality to hold. 

## K-Fold Cross Validation

This approach involves randomly dividing the set of observations into k groups or folds, of approximate the same size. 

The first fold is treated as a validation set and the method is fitted on the remaining k-1 folds. The $MSE_1$ is then computed on the observations of the held out fold 1. 

We repeat k times, each time a different group of observation is treated as a validation set. This process results in k estimates of the test error. the k-fold CV estimate is computed by averaging these values :

$$
CV_{k} = \frac{1}{k} \sum_i MSE_i
$$

It is not hard to see that LOOCV is a special case of k-fold where k = n

In practice, one typically performs k-fold CV using k=5 or k=10. The most obvious advantage is computational. some statistical methods have computationally intensive fitting procedures, so LOOCV may be computationally heavy espeically for large n. fitting using 10-fold CV is mouch more feasible.

![Fig 5.5 from G James et al](images/Screenshot%202024-07-21%20Figure5.5%20G%20James.png)

There may be some variablity in the CV estimates as a result of how observations are divided into 10 folds. But this variability is much lower than the variability is typically much lower than the variability in the test error estimates that results from validation set approach. 

When we examine real data, we do not know the true test MSE, and
so it is difficult to determine the accuracy of the cross-validation estimate.
However, if we examine simulated data, then we can compute the true
test MSE, and can thereby evaluate the accuracy of our cross-validation
results. 

When we perform cross-validation, our goal might be to determine how well a statisictial learning procedure can be expected to perform on indepedent data - then MSE will be of interest

In other times, we may be interested only in the location of the min point in the estimated test MSE curve. We may be performing cross validation on a number of statistical learning methods or a single method with different flexibility - for the ultimate purpose of identifying the method that results in the lowest test error.
here, location of min point on MSE curve is more important than the MSE. 

### Bias Variance trade off for k-fold Cross validation

One important advantage of k-fold CV is that it often gives more accurate estimates of the test error rate than LOOCV.

- LOOCV gives approximately unbiased estimates of test error, since each training set contains n-1 observations.
- Performing K-fold CV will lead to an intermediate level of bias since each training set contains approximately (k-1)n/k observations. In terms of bias reduction, in may seem that LOOCV is preferred to.

However, it turns out that LOOCV has higher variance than k-fold does. with k < n
- when we perform LOOCV, we are averaging the outputs of n-fitted models, each of which is trained on an almost identitical set of observations. ie these outputs are highly correlated with each other.
- In contrast, K-fold CV averages the k-fitted models that are somewhat less correlated with each other as the overlap between training sets are smaller. 
- Since the mean of many highly correlated quantities has higher variance, the test error estimate resulting from LOOCV tends to have higher variance than does the test error estimate from k-fold CV. 

## Classification case

We apply the same concepts, even if Y is qualitative. instead of MSE, we instead use the number of misclassified observations where $y_i \ne \hat{y_i}$

To select between models, we can use CV to make the decision. We can compare error rate against order of polynomial used , or error rate against parameter (eg K in KNN)

# Bootstrap

It can be easily applied to a wide range of statistical learning methods, including some for which a measure of variablity is otherwise difficult to obtain and not automatatically output by software. 

E.g. we repeat the process of simulation, to get maybe 1000 estimates of the parameter of interest - SE of $\alpha$ in $\alpha + \beta x = y$

The bootstrap approach allows us to use a computer to emulate the process of obtaining new sample sets, so that we can estimate the variability of interest. 
- rather than repeatedly obtaining independent data sets from the population, we instead obtain distinct data sets by repeatedly sampling observations from the original data set. 

example: we have n=3 observations. 

- We randomly select n observations from the data set in order to produce a bootstrap data set, $Z*^1$ The samplingis done with replacment, the same observatin can occure more thanonce in the bootstrap data. ($Z*^1$ contains 1 of obsv 1 and 2 of obsv 3 but no obsv 2)
- We use $Z*^1$  to produce a bootstrap estimate for $\alpha$ which we denote as $\alpha*^1$. This procedure is repeated B times for some large B to produce different bootstrap data sets and estimates. 
- We compute the Standard error of the estimates using :

$$
SE_B(\hat{\alpha}) = \sqrt{\frac{1}{B-1}\sum_r^B (\hat{\alpha*^r} - \frac{1}{B}\sum_{r'}^B\hat{{\alpha*^{r'}}})^2}
$$

![Fig 5.11 from G James et al](images/Screenshot%202024-07-23%20Fig5.11%20G%20James.png)
