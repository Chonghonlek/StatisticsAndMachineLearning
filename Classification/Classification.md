Notes taken from  **Gareth J et al., An introduction to Statistical Learning**

## Classification

Compared to linear regression, we are trying to predict qualitative response. 

Linear regression is not suitable as it cannot accomodate a qualtitative response with more than 2 cases eg

$$
Y =
\begin{cases}
1 &\text{if a}\\
2 &\text{if b}\\
3 &\text{if c}\\
\end{cases} 
$$

It also does not provide meaning estimates of $P(Y|X)$ even with just 2 classes. Below, we see that some estimates under linear regression go beyond $[0,1]$

![figure 4.2 Taken from Gareth J et al](../images/Screenshot%202024-05-29%20Figure4.2%20G%20James.png)

## Comparison across Classification methods

We assign an observation to the class that maximises $P(Y=k|X=x)$. This is equivalent to setting the K-th Class as the baseline class and assign an observation to the class that maximises: $\log \frac{P(Y=k|X=x)}{P(Y=K|X=x)}$ for $k =1, \cdots , K$

### LDA

We make use of bayes theorem and the assumption that the predictors within each class are drawn from multivariate normal density with class specific mean and shared covariance matrix  to show that:

$$
\begin{aligned}
    \log \frac{P(Y=k|X=x)}{P(Y=K|X=x)} &= \log \frac{\pi_k f_k(x)}{\pi_K f_K(x)} \\
    &= \log \frac{\pi_k \exp (-\frac{1}{2}(x-\mu_k)^T\Sigma^{-1}(x-\mu_k))}{\pi_K \exp (-\frac{1}{2}(x-\mu_K)^T\Sigma^{-1}(x-\mu_K))} \\
    &= \log \frac{\pi_k }{\pi_K} -\frac{1}{2}(x-\mu_k)^T\Sigma^{-1}(x-\mu_k) + \frac{1}{2}(x-\mu_K)^T\Sigma^{-1}(x-\mu_K) \\
    &= \log \frac{\pi_k }{\pi_K} -\frac{1}{2}(\mu_k+\mu_K)^T\Sigma^{-1}(\mu_k-\mu_K) + x^T\Sigma^{-1}(\mu_k-\mu_K)\\
    &= a_k + \sum_{j=1} b_{kj}x_j
\end{aligned}
$$

where $a_k = \log \frac{\pi_k }{\pi_K} -\frac{1}{2}(\mu_k+\mu_K)^T\Sigma^{-1}(\mu_k-\mu_K)$ and $b_{kj}$ is the j-th component of $\Sigma^{-1}(\mu_k-\mu_K)$

This is similar to logistic regression that assumes log odds of the posterior probabilities is linear in x. recall for logistic regression $\log \frac{P(Y=k|X=x)}{P(Y=K|X=x)} = \beta_{k0} + \beta_{k1}X_1 + \cdots \beta_{kp}X_p$

### QDA 

$$
\log \frac{P(Y=k|X=x)}{P(Y=K|X=x)} = a_k + \sum_{j=1} b_{kj}x_j + \sum_j \sum_l c_{kjl}x_jx_l
$$

QDA assumes the log odds of the posterior is quadratic in x

### Naive Bayes

$$
\begin{aligned}
        \log \frac{P(Y=k|X=x)}{P(Y=K|X=x)} &= \log \frac{\pi_k f_k(x)}{\pi_K f_K(x)} \\
        &= \log \frac{\pi_k\prod_j f_{kj}(x_j)}{\pi_K \prod_j f_{Kj}(x_j)} \\
        &= \log \frac{\pi_k }{\pi_K} + \sum_j \log \frac{ f_{kj}(x_j)}{ f_{Kj}(x_j)} \\
        &= a_k + \sum_{j=1} g_{kj}(x_j)
\end{aligned}
$$

The right hand side takes the form of a generalised additive model (To be discussed)

### Observations

- LDA is a special case of QDA with $c_{kjl} = 0$, which is not surprising since LDA is a restricted version of QDA with all classes sharing same covariance matrix
- Any classifier with a linear decision boundary is a special case of naive bayes with $g_{kj}(x_j) = b_{kj}x_j$. LDA is a special case of naive Bayes. This is not obvious given that both method make different assumptions: LDA assumes features are normally distributed with a common within-class covarinace matrix, and Naive Bayes assumes independence of the features
- if we model $f_{kj}(x_j)$ in the naive bayes classifier with a 1-d gaussian distribution then we end up with $g_{kj}(x_j) = b_{kj}x_j$ where $b_{kj} = (\mu_{kj} - \mu_{Kj})/\sigma^2_j$. In this case, naive Bayes is actually a special case of LDA with $\Sigma$ restricted to be a diagonal matrix with the jth diagonal element equal to $\sigma^2_j$
- Neither QDA nor naive bayes is a special case of the other. 

None of these methods uniformly dominates the others. In any setting, the choice of method will depend on the true distribution of the predictors in each of the K classes as well was other values $n,p$. The latter ties into the bias-variance trade off

- Comparing logistic regression to LDA, we expect LDA to outperform logistic regression when the normality assumption holds, vice versa. 

### KNN or K-Nearest Neighbours (Discussed in StatsLearning.md)

- Comparing to KNN, KNN is a non -parametric approach that makes a prediction for an observation by taking the training observations that are closest to x and assigning to the class to whcih the plurality of these observations belong. 

- This approach dominates LDA and logistic regression when the decision boundary is highly non linear provided that n is very large and p is small. 
- In order to provide accurate classification, KNN requires a lot of observations relative to the number or predictors - that is n >> p. this has to do with the fact that KNN is non parametric and thus tends to reduce bias while incurring alot of variance
- In settings where the decision boundary is non-linear but n is only modest or p is not very small, then QDA is preferred to KNN. This is because QDA can provide a non -linear decsion boundary while taking advantage of a parametric form - it requires a smaller sample size for accurate classification relative to KNN. 

| Syntax      | Description | Remarks|
| ----------- | ----------- |---|
| Small n / Small p  (higher variance)    | LDA/Naive Bayes  | Naive Bayes reduce variance more
| Bigger n   | QDA/KNN        | QDA may work better than KNN at a slight lower sample size

### Refer to textbook for emperical evidence - binary prediction example with 100 random training data sets for each scenario

When the true decision boundaries are linear, then the LDA and logistic regression approaches will tend to perform well. When the boundaries are moderately non-linear, QDA or naive Bayes may give better results. Finally, for much more complicated decision boundaries, a non-parametric approach such as KNN can be superior. But the level of smoothness for a non-parametric approach must be chosen carefully.

- Scenario 1 (linear) : 20 training observations in each of the 2 classes. Observations within each class were uncorellated random normal variables with a different mean in each class. LDA and logistic regression performed well as it assumes linear decision boundary. KNN -1 performed poor in terms of variances that is not offset by a reduction in bias. QDA performed worse than LDA as it fit a more flexible classifier than necessary. The performance of naive bayes was slightly better than QDA as naive bayes assumption of indepdent predictors is correct. 

- Scenario 2 (linear) Now, the 2 predictors had correlation of -0.5. This caused naive bayes to perform poorly given that the assumption of independet predictors is violated.

- Scenario 3 (linear). Now the negative correlation still remains. But this time we generate $X_1$ and $X_2$ from t-distribution - which generated more extreme points. This set up violayed the assumptions of LDA since the observations were not drawn from a normal distribution. Logistic regression performed better than LDA. QDA also detiorated given the non-normality and naive bayes also performed poorly given the independence assumption

- Scenario 4 (non-linear) Data is now generated from normal distribution with correlation of 0.5 between predictors in first class and correlation of -0.5 between predictors in the 2nd class. This corresponds to QDA assumption and resulted in quadratic decision boundaries. QDA outperformed all other approaches. Naive bayes assumption violated. 

- Scenario 5 (non-linear) Data is generated from normal distribution with uncorrelated predictors. KNN-cross validation method gave best result. But KNN-1 was worst. this implies that even when data exhibits a complex non-linear relationship, a non parametric method such as KNN can still give poor results if level of smoothness is not chosen correctly. 

- Scenario 6 (non-linear) now, observations come from normal distribution with a different diagonal covarinace matrix for each class. but now sample size is very small. Naive bayes performed well as assumptions were met. LDA and logistic regression performed poorly because of the nonlinear decision boundary. QDA performed poorly to naive bayes given the very small size. KNN also suffered due to very small sample size. 

--- 

