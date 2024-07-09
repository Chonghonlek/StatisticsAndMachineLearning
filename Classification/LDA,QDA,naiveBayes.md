Notes taken from  **Gareth J et al., An introduction to Statistical Learning**

Consider a less direct approach to estimate the probabilities $P(Y=k|X=x)$, we model the distrubution of the predictors X separately in each of the response classes (for each value of Y) -> $P(X=x|Y=k)$

We then use **bayes theorem** to flip and estimate $P(Y=k|X=x)$. When the distribution of X is normal within each class, it turns out the model is very similar to the logistic regression model.

We use this method when :

1) There is substanrial separation between the 2 classes, the parameter estimates for the logistic regression model are surprisingly unstable
2) If distribution of predictors X is approximately normal in each of the classes and the sample size is small, then this approach is more accurate then logistic regression
3) This method can be extended to the case of more than 2 response classes. (we also can use multinomial logistic regression)
   
In this model, we let $\pi_k$ represent the prior for observation of the kth class  - P(y = k) . We then let $f_k(X) = P(X|Y=k)$ be the density of X that comes from kth class(likelihood function). _Bayes' theorem_ states that the posterior probability is :

$$
P(Y=k|X=x) = \frac{\pi_k f_k(X)}{\sum_{l=1}^{k}\pi_l f_l(X)} \quad \text{recall that Y is catogorical, so we use the discrete cases}
$$

We can plug in estimates of $\pi_k$ and $f_k(x)$ to get the posterior. but getting the estimate of $f_k(x)$ is challenging. If We can find a way to estimate $f_k(x)$ , we can approximate the bayes classifer - classify observation x to which $P(Y=k|X=x)$ is the largest.

## Bayes classifier for p = 1

Assume we only have one predictor p = 1 or number of x-variables = 1. We would like to obtain an estimate for $f_k(x)$ that we can plug into to estimate $p_k(x)$ - posterior. we then classify an observation to a class where $p_k(x)$ is the greatest

in particular, we assume $f_k(x)$ - likelihood to be gaussian or normal. in 1-d setting, $f_k(x) = \frac{1}{\sqrt{2\pi}\sigma_k} \exp (\frac{-1}{2\sigma^2_k} (x-\mu_k)^2 )$, where $\mu_k$ and $\sigma^2_k$ are the mean and variance parameter for the kth class. 

Suppose the variance term across all K classes is the same, i.e. $\sigma_1^2 = \cdots = \sigma_k^2$ . We can estimate posterior with:

$$
p_k(x) = P(Y=k|X=x) = \frac{\pi_k \frac{1}{\sqrt{2\pi}\sigma} \exp (\frac{-1}{2\sigma^2} (x-\mu_k)^2 )}{\sum_{l=1}^{k}\pi_l \frac{1}{\sqrt{2\pi}\sigma} \exp (\frac{-1}{2\sigma^2} (x-\mu_l)^2)}
$$

By taking log on both sides,

$$
\begin{aligned}
\log(p_k(x)) &= \log\pi_k  + \frac{-1}{2\sigma^2}(x-\mu_k)^2 - \log(\sum_{l=1}^{k}\pi_l \frac{1}{\sqrt{2\pi}\sigma} \exp (\frac{-1}{2\sigma^2} (x-\mu_l)^2)) \\
& = \log\pi_k - \frac{x^2}{2\sigma^2} + x \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \cdots \\
& = \log\pi_k + x \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} - A
\end{aligned}
$$

This is equivalent to assignining the observation to the class for which $\delta_k(x)  = \log\pi_k + x \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2}$ is the maximum.

For instance K = 2 and $\pi_1 = \pi_2$, then the bayes classifier assignas an observation to class 1 if :

$$
\begin{aligned}
\log\pi_1 + x \frac{\mu_1}{\sigma^2} - \frac{\mu_1^2}{2\sigma^2} &> \log\pi_2 + x \frac{\mu_2}{\sigma^2} - \frac{\mu_2^2}{2\sigma^2} \\
x (\mu_1 - \mu_2) &> \frac{\mu_1^2 - \mu_2^2}{2} \\
2x(\mu_1 - \mu_2) &>\mu_1^2 - \mu_2^2
\end{aligned}
$$

The bayes boundary is the point for which $\delta_1(x) = \delta_2(x) \quad \iff x = \frac{\mu_1^2 - \mu_2^2}{2(\mu_1 - \mu_2)} = \frac{\mu_1 + \mu_2}{2}$

Given the parameters of a normal density function and assuming equal observation, we can compute the bayes classifier. However, in real life, we cant calculate this

## Linear Discriminant Analysis (LDA) for p = 1 (number of x variables = 1)

The LDA approximates the Bayes classifier by plugging estimates for $\pi_k,\mu_k,\sigma^2$ into $\delta_k(x)$

We use the following estimates, where n is the total number of observation, and $n_k$ is the number if training observation in the kth class:

$$
\begin{aligned}
\hat{\mu_k} &= \frac{1}{n_k}\sum_{i:y_i = k} x_i \\
\hat{\sigma^2} &= \frac{1}{n-K} \sum_{k=1}^K \sum_{i:y_i = k} (x_i - \hat{\mu_k})^2
\end{aligned}
$$

$\hat{\mu_k}$ is the average of the training observations of the k-th class and $\hat{\sigma^2}$ is the weighted average of the sample variances for each of the K classes.

In the absence of information on the class probabilities, LDA estimates the $\pi_k$ using the  proportion of the training observations that belong to the k-th class. i.e. $\hat{\pi_k} = n_k / n$

The LDA classifier plugs the estimates given and assigns an observation X=x to the class where $\hat{\delta_k(x)} = \log\hat{\pi_k} + x \frac{\hat{\mu_k}}{\hat{\sigma^2}} - \frac{\hat{\mu_k^2}}{2\hat{\sigma^2}}$ is the largest

The word linear stems from the fact that the disciminant function $\hat{\delta}_k(x)$ are linear functions of x. 

LDA classifier results from assuming the observations within each class come from a normal distribution w specified parameters and estimating these parameter. 

![figure 4.4 from G James et al](../images/Screenshot%202024-06-10%20Figure4.4%20G%20James.png)

## Linear Discriminant Analysis (LDA) for p > 1 (number of x variables > 1)

Suppose now we have $X = (X_1,X_2,\cdots,X_p)$ which is drawn from a multivariate Gaussian/Normal Distribution with a class-specific mean vector and common covariance matrix. Recall that if X follows a multivariate normal dist i.e $X \sim N(\mu,\Sigma)$ where E(X) = $\mu$ and Cov(X) = $\Sigma$ and p is the number of dimensions of X, then the pdf of X: 

$$
P(X=x) = \frac{1}{(2\pi)^{p/2} * det(\Sigma)^{1/2}} \exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))
$$

For the case where p>1, the LDA classifier assumes the observation in the kth class are drawn from a multivariate Gaussian distribution. in fact, plugging the density into bayes formula we can deduce that the Bayes classifier assigns an observation X=x for which :

$$
\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + log (\pi_k)
$$

is the largest. the matrix version of p=1 case.

![figure 4.6 from G James et al](../images/Screenshot%202024-06-10%20Figure4.6%20G%20James.png)

From above, the dash lines that belong to the bayes decision boundaries is the set for which : $\delta_k(x) = \delta_l(x)$ for $k \ne l$ i.e. $x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k = x^T\Sigma^{-1}\mu_l - \frac{1}{2}\mu_l^T\Sigma^{-1}\mu_l$

we estimate the unknown parameters : $\mu_1,\dotsc,\mu_k,\pi_1,\dotsc,\pi_k \text{ and } \Sigma$

LDA plugs these estimates to obtain quantities $\hat{\delta}_k$ and classifies the observation for which $\hat{\delta}_k$ is the largest.

### Example of LDA

Suppose we perform LDA predictions to model default statuses based on predictors: credit card balance(quantitative) and student status(qualitative - model assumption violated).

![table 4.4 from G james et al](../images/Screenshot%202024-06-10%20Table4.4%20G%20James.png
)

from the **confusion matrix**, We see that LDA model fit to the 10000 training samples result in a training error rate of  (23+252)/10000 = 2.75%

we note 2 caveats:

1) Training error rates will be lower than test error rates since we specifically adjust parameters of our model to do well for training data. the higher the ratio of parameters to number of samples, the more we expect overfitting to play a role

2) Only 3.33% in training sample defaulted. A classifier that predicts only non default will result in error rate of 3.33% as well. this is at most a bit higher than the LDA training error rate!!

Also, while only 23 out of 9667 individuals who did not default was incorrectly labeled, 252 of the 333 individuals who defaulted were missed by LDA. Even though overall error rate is low, the error rate among individuals who defaulted is very high (252/333 = 75.7%)

in the above case, the sensitivity is the percentage of true defaulters identified which is 24.3% (81/333) and the specificity is the percentage of non-defaulters identified which is 99.8% (1 - 23/9667)

Recall that **Sensitivity  = True positive rate, Specificity = True Negative rate**

We must understand that LDA is trying to approximate the Bayes' classifier and that the Bayes classifier will try to get the smallest possible total number of misclassifications regardless of the class!

We can modify LDA such that, for a class K=2, we adjust the threshold eg $P(\text{default}=\text{Yes}|X=x) > 0.2$ instead of 0.5. We adjust the probability based on the context.

![Table 4.5 and Figure 4.7 from G james et al](../images/Screenshot%202024-06-10%20Table4.5&Figure4.7%20G%20James.png)

In fact, this method improves the sensitivity to 41.4% but resulted in a higher overall error rate to 3.73%.

NOTE: The blue line above represents **(1 - sensitivity) = False negative rate = Type II error**

### Assessing Performance

![Figure 4.8 from G james et al](../images/Screenshot%202024-06-10%20Figure4.8%20G%20James.png)

The above displays the **ROC (receiver operating characteristics) curve. x-axis is (1 - specificity) = Type I error, y-axis = sensitivity**

The overall performance is given by the **area under the curve (AUC)**. 

An ideal ROC curve will hug the top left corner, so the larger the AUC the better the classifier. 

For the default example, AUC is 0.95 which is considired very good. 

Below is a summary of performance measures:

![Table 4.6 and 4.7 from G james et al](../images/Screenshot%202024-06-10%20Table4.6&4.7%20G%20James.png)

Recall that type I error is rejected when $H_0$ true and type II error is accepted when not true($H_1$)


## Quadraric Discriminant Analysis (QDA)

Unlike LDA, QDA classifier assumes that each class has its own covariance matrix i.e. for the kth class $X \sim N(\mu_k,\Sigma_k)$

Under this assumption the bayes classifier assignas an observation $X=x$ to the class where:

$$
\begin{aligned}
\delta_k(x) &= -\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k) -\frac{1}{2}log(det(\Sigma_k)) + log\pi_k \\
&= -\frac{1}{2}x^T\Sigma_k^{-1}x + x^T\Sigma_k^{-1}\mu_k -\frac{1}{2}\mu_k^T\Sigma_k^{-1}\mu_k-\frac{1}{2}log(det(\Sigma_k)) + log\pi_k
\end{aligned}
$$

is the largest. (this function is a quadratic function of x)

QDA estimates $\pi_k,\mu_k,\Sigma_k$ and plugs them above and assigns observation x to class where $\hat{\delta}_k$ is the largest. 

----
### Why we may prefer LDA or QDA

We go back to the bias-variance trade off. For p predictors,
- LDA estimates the covariance matrix that requires ${p\choose2} = p(p+1)/2$ parameters
- QDA requires each kth class to have its own covariance matrix and thus estimates $K * p(p+1)/2$ parameters

Consequently this means that LDA is much less flexible classifier than QDA and has thus lower variance - that allows it to perform better with new data set. 

However, if the assumption that K classes share the same covariance matrix is very off, then LDA suffers from high bias

Typically,
- LDA performs for low training observations and thus reducing variance is crucial
- QDA is recommended for large traininig data set, so variance is not a concern AND if the assumption of common covariance matrix is untenable.

![Figure 4.9 from G james et al](../images/Screenshot%202024-07-08%20Figure4.9%20G%20James.png)

## Naive Bayes

Instead of assuming that the function $f_k(x)$ - likelihood belongs to a particular family of distrutions. We assume that :
- within the kth class, the p predictors are independent i.e. $f_k(x) = f_{k1}(x_1) \times \cdots \times f_{kp}(x_p)$

By doing this, we eliminate the need to worry about the association between the p predictors. 

In most settings, we do not believe that the naive bayes assumption that p covariates are indepedent within each class. but even though this modelling assumption is made for convenience , it often leads to pretty decent results. Especially in settings where n is not large enough relative to p for us to effectively estimate the joint distribution of predictors within each class.

The expression for the posterior probability for $k = 1\cdots K$ :

$$
P(Y=k|X=x) = \frac{\pi_k f_{k1}(x_1) \times \cdots \times f_{kp}(x_p)}{\sum_{l=1}^{k}\pi_l  f_{l1}(x_1) \times \cdots \times f_{lp}(x_p)}
$$

To estimate the one-dimensional density of $f_kj$ (likelihood for each predictor in the kth class) using data (j predictors) $x_{1j},\dotsc x_{nj}$, we can:

1) if $X_j$ is quantitative, assume $X_j|Y = k \sim N(\mu_{jk},\sigma^2_{jk})$. This means that within each class the jth predictor is drawn for normal distribution. However there is a further assumption that predictors are independent and this amounts to QDA with a class-specific covariance matrix that is diagonal. 

2) if $X_j$ is quantitative, we use a non-parametric estimate for $f_kj$. one way to make a historgram for all the observations of the jth predictor for each class then estimate  $f_kj(x_j)$ as a fraction of the training observations in the kth class that belong to the same histogram bin as $x_j$. Alternatively we use a kernel density estimator. 

3) if $X_j$ is qualitative, we simply count the proportion of training observations for the jth predictor corresponding to each class. E.g. if $X_j \in {1,2,3}$ and we have 100 observations for the kth class. then we take the number of occurences of ${1,2,3}$ and divide by 100 to estimate the likelihood. 

### Example from textbook

We now consider the naive Bayes classifier in a toy example with p = 3
predictors and K = 2 classes. The first two predictors are quantitative,
and the third predictor is qualitative with three levels. Suppose further
that $\hat{\pi_1} = \hat{\pi_2}= 0.5$ 

Now suppose that we wish to classify a new observation, $x^∗ = (0.4, 1.5, 1)^T$. It turns out that in this example, $\hat{f_{11}}(0.4) = 0.368, \hat{f_{12}}(1.5) = 0.484, \hat{f_{13}}(1) = 0.226$, and $\hat{f_{21}}(0.4) =0.030, \hat{f_{22}}(1.5) = 0.130, \hat{f_{23}}(1) = 0.616$. Plugging these estimates results in posterior probability estimates of $Pr(Y = 1|X = x^∗) = 0.944$ and $Pr(Y = 2|X = x^∗) = 0.056$.

In this example, it should not be too surprising that naive Bayes does not convincingly outperform LDA: this data set has n = 10,000 and p = 4,and so the **reduction in variance resulting from the naive Bayes assumption is not necessarily worthwhile**. 

We expect to see a greater pay-off to using naive Bayes relative to LDA or QDA in instances where **p is larger or n is smaller**, so that reducing the variance is very important
