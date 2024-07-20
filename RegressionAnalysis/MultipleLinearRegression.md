Notes taken from **Douglas C M et al, Introduction to Linear Regression Analysis**

# Multiple linear regression

A regression model with more than 1 regressor variable is called multiple regression model. 

We consider the model with k regressors (p = k+1 coefficients) : $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... \beta_kx_k + \epsilon$

Assume error term $\epsilon$ has $E(\epsilon) = 0$ and $Var(\epsilon) = \sigma^2$ and that their errors are uncorrelated

We use least squares estimation to get the regression coefficient

## Least square estimation

$$
\text{let S } = \sum \epsilon_i^2 = \sum (y_i - \beta_0 - \sum \beta_j X_j)^2
$$

Then we minimise the function S with respect to $\beta_0 , \beta_1 , \beta_2,\dotsc,\beta_k$. 

The least squares estimator must satisfy :

$$
\frac{\partial S}{\partial \beta_0} \bigg|_{\hat{\beta}_0 , \hat{\beta}_1 , \hat{\beta}_2,\dotsc,\hat{\beta}_k} = -2 \sum_{i=1}^n (y_i - \hat{\beta}_0 - \sum_j \hat{\beta}_j x_{ij}) = 0
$$

$$
\frac{\partial S}{\partial \beta_j} \bigg|_{\hat{\beta}_0 , \hat{\beta}_1 , \hat{\beta}_2,\dotsc,\hat{\beta}_k} = -2 \sum_{i=1}^n (y_i - \hat{\beta}_0 - \sum_j \hat{\beta}_j x_{ij})x_{ij} = 0 \quad \text{for j=} 1,\dotsc k
$$

We obtain the least squares normal equations

$$
\begin{aligned}
&n \hat{\beta}_0 + \hat{\beta}_1 \sum_i x_{i1} + \hat{\beta}_2 \sum_i x_{i2} + \cdots \hat{\beta}_k \sum_i x_{k1} = \sum_i y_i \\
&\hat{\beta}_0 \sum_i x_{i1} + \hat{\beta}_1 \sum_i x_{i1}^2 + \hat{\beta}_2 \sum_i x_{i1}x_{i2} + \cdots \hat{\beta}_k \sum_i x_{i1}x_{ik} = \sum_i x_{i1}y_i
\\
\vdots \\
\\
&\hat{\beta}_0 \sum_i x_{ik} + \hat{\beta}_1 \sum_i x_{i1}x_{ik} + \hat{\beta}_2 \sum_i x_{ik}x_{i2} + \cdots \hat{\beta}_k \sum_i x_{ik}^2 = \sum_i x_{ik}y_i
\end{aligned}
$$

It is more convenient to deal with this using matrix notation: $y = X \beta + \epsilon$

$$
\begin{aligned}
&y =
\begin{pmatrix}
y_1\\y_2\\ \vdots \\ y_n
\end{pmatrix}, \quad && X = 
\begin{pmatrix}
1 & x_{11} &\cdots & x_{1k}\\
1 & x_{21} &\cdots & x_{2k}\\
\vdots & \vdots & &\vdots\\
1 & x_{n1} &\cdots & x_{nk}
\end{pmatrix} \\
&\beta = 
\begin{pmatrix}
\beta_1\\ \beta_2\\ \vdots \\ \beta_k
\end{pmatrix}, \quad && \epsilon = 
\begin{pmatrix}
\epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n
\end{pmatrix}
\end{aligned}
$$

$$
\begin{pmatrix}
y_1\\y_2\\ \vdots \\ y_n
\end{pmatrix} =
\begin{pmatrix}
1 & x_{11} &\cdots & x_{1k}\\
1 & x_{21} &\cdots & x_{2k}\\
\vdots & \vdots & &\vdots\\
1 & x_{n1} &\cdots & x_{nk}
\end{pmatrix}
\begin{pmatrix}
\beta_1\\ \beta_2\\ \vdots \\ \beta_k
\end{pmatrix} + 
\begin{pmatrix}
\epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n
\end{pmatrix}
$$

We now proceed to find the vector of least squares estimatators $\hat{\beta}$

$$
\begin{aligned}
\text{Let } S(\beta) &= \sum \epsilon_i^2 = \epsilon'\epsilon = (y-X\beta)'(y-X\beta) \\
&= (y' - \beta'X')(y-X\beta) = y'y - y'X\beta - \beta'X'y + \beta'X'X\beta \\
&= y'y - 2\beta'X'y + \beta'X'X\beta \quad \because y'X\beta \text{ is scalar} \implies y'X\beta = \beta'X'y\\
\end{aligned}
$$

Consider the lemma $\frac{\partial a'B}{\partial B} = a$ and $\frac{\partial B'AB}{\partial B} = 2AB$, then the least squares estimator must satisfy the following

$$
\begin{aligned}
\frac{\partial S}{\partial B}\bigg|_{\hat{\beta}} &= -2X'y + 2X'X\hat{\beta} = 0 \\
&\implies X'X\hat{\beta} = X'y\\
\begin{pmatrix}
n & \sum x_{i1} &\cdots & \sum x_{ik}\\
\sum x_{i1} & \sum x_{i1}^2 &\cdots & \sum x_{i1}x_{ik}\\
\vdots & \vdots & &\vdots\\
\sum x_{ik} & \sum x_{ik}x_{i1} &\cdots & \sum x_{ik}^2
\end{pmatrix}
&\begin{pmatrix}
\hat{\beta_1}\\ \hat{\beta_2}\\ \vdots \\ \hat{\beta_k}
\end{pmatrix} = 
\begin{pmatrix}
\sum y_i \\ \sum x_{i1}y_i\\ \vdots \\ \sum x_{ik}y_i
\end{pmatrix} \\
\text{from here,}&\implies \hat{\beta} = (X'X)^{-1}X'y
\end{aligned}
$$

The least squares estimator of $\beta$ is $\hat{\beta} = (X'X)^{-1}X'y$, we use this r-code to get estimator $\hat{\beta}$

```r
beta_hat = solve(XTX,XTy)
```

The fitted regression model corresponding levels to the regressor variables 
$x' = [1,x_1,\cdots,x_k]$ is $\hat{y} = x'\hat{\beta}$

The vector $\hat{y} = X\hat{\beta} = X(X'X)^{-1}X'y = Hy$, here $H = X(X'X)^{-1}X'$ is called the **hat matrix**, it maps the vector of observed values into a fitted value. Diagonals of the hat matrix is called hat values.

The difference between the observed value $y_i$ and fitted value $\hat{y_i}$ is called the **residual**

We can express the residual as $e = y - X\hat{\beta} = y - Hy = (I-H)y$

### Geometric interpretation of least squares

![Figure 3.6 from Douglas M et al](../images/Screenshot%202024-07-01%20Fig3.6%20D%20Montgomery.png)

Sample space is 3-D in the above figure. We attempt to minimise the distance between point B and A. The squared distance is a minimum when the point is the foot of the line from A normal to the estimation space. 

## Properties of Least Squares Estimators

Recall, $\hat{\beta} = (X'X)^{-1}X'y$

$$
\begin{aligned}
E(\hat{\beta}) &= E((X'X)^{-1}X'y) = E((X'X)^{-1}X'(X\beta + \epsilon)) \\
&= E((X'X)^{-1}X'X\beta + (X'X)^{-1}X'\epsilon) = E(\beta +(X'X)^{-1}X'\epsilon ) \\
&= \beta +(X'X)^{-1}X'E(\epsilon) = \beta
\end{aligned}
$$

We can see that $\hat{\beta}$ is an unbiased estimator of $\beta$ if the model is correct

----

The Variance property of $\hat{\beta}$ is expressed via the covariance matrix. $Cov(\beta) = E[(\hat{\beta} - E(\hat{\beta}))(\hat{\beta} - E(\hat{\beta}))']$

The **covariance matrix** is a p x p symmetric matrix , whose j-th diagonal element is the variance of $\hat{\beta}_j$ and ij-th off-diagonal element is the covariance between $\hat{\beta}_i$ and $\hat{\beta}_j$.

To find the covariance matrix, we apply variance operator to $\hat{\beta}$

Note that we use the following lemma: $E(Ay) = A\mu$ , and $Var(Ay) = AVA'$

$$
\begin{aligned}
Var(\hat{\beta}) &= Var((X'X)^{-1}X'y) = (X'X)^{-1}X' Var(y)[(X'X)^{-1}X']' \quad \because\text{Var(Ay) = AVA'} \\
&=(X'X)^{-1}X' (\sigma^2I)[X(X'X)^{-1}] \quad \text{Using property of transpose} \\
&= \sigma^2 (X'X)^{-1}X'X(X'X)^{-1} \\
&= \sigma^2 (X'X)^{-1} 
\end{aligned}
$$

If we let $C=(X'X)^{-1}$, then $Var(\hat{\beta}_j) = \sigma^2 C_{jj}$ and $Cov(\hat{\beta}_i,\hat{\beta}_j)  = \sigma^2 C_{ij}$

infact, The Gauss-Markov Theorem (not discussed here) establishes that the least squares estimator $\hat{\beta}$ is the best linear unbiased estimator of $\beta$. Also, if we further assume error sare normally distributed then the  $\hat{\beta}$ is also the MLE of $\beta$. the MLE is the minimum variance unbiased estimator of $\beta$

### estimating $\sigma^2$

Recall that $SS_{res} = \sum (y_i - \hat{y}_i)^2 = \sum e_i^2 = e'e$

$$
\begin{aligned}
SS_{res} &= e'e =(y-X\hat{\beta})'(y-X\hat{\beta}) \\
&= (y'-\hat{\beta}'X')(y-X\hat{\beta}) \\
&= y'y - \hat{\beta}'X'y - y'X\hat{\beta} + \hat{\beta}'X'X\hat{\beta} \\
&= y'y - 2\hat{\beta}'X'y + \hat{\beta}'X'X\hat{\beta}\\
&= y'y - \hat{\beta}'X'y \quad \because X'X\hat{\beta} = X'y \text{ from least squares estimation}
\end{aligned}
$$

(proof not shown) The residual sum of square has n-p degree of freedom and thus the residual mean square or $MS_{res} = \frac{SS_{res}}{n-p}$

(proof not shown) The expected value of $MS_{res} = \sigma^2$, thus an unbiased estimator of $\sigma^2 = MS_{res}$. We note that this estimator is model dependent. 

**We usually prefer a model with small residual mean square**

## Hypothesis testing 

How do we know the overall adequacy of the model? which regressors are the more important ones?

We use hypothesis testing to address them. These test require us to assume that our random errors be independent and follow normal dist N(0,1)

### Testing the significance of Regression

We test $H_0 : \beta_1 = \cdots = \beta_k = 0$ against $H_1: \beta_j \ne 0$ for at least one j. 

The test procedure is a generalization of the ANOVA test. Recall that $SS_T = SS_R + SS_{res}$ and that $\frac{SS_R}{\sigma^2} \sim \chi^2(k)$ and $\frac{SS_Res}{\sigma^2} \sim \chi^2(n-k-1)$ (proof not shown)

Then we have that $F_0 = \frac{SS_R/k}{SS_Res/(n-k-1)} = \frac{MS_R}{MS_{res}}$ folows the $F_{k,n-k-1}$ distribution

If $H_0$ is true, $\frac{SS_R}{SS_{res}} \rightarrow 0$.

We reject $H_0$  when $F_0 > F_\alpha(k,n-k-1)$

note that:

$$
\begin{aligned}
SS_{res} &= y'y - \hat{\beta}'X'y \\
y'y &= SS_{res} + \hat{\beta}'X'y \\
\sum y_i^2 - n\bar{y}^2 &= SS_{res} + \hat{\beta}'X'y - n\bar{y}^2 \\
SS_T &= SS_{res} + \hat{\beta}'X'y - n\bar{y}^2 \\
\implies SS_R &= \hat{\beta}'X'y - n\bar{y}^2 \\
\end{aligned}
$$

---

### $R^2$ and adjusted-$R^2$

Another way to assess the overall adequacy of the model are the  $R^2$ and adjusted-$R^2$ 

Recall that $R^2 = \frac{SS_R}{SS_T} = 1- \frac{SS_{res}}{SS_T}$

We have that adj-$R^2 = 1- \frac{SS_{res}/(n-p)}{SS_T/(n-1)}$ 

$R^2$ increases when regressor is added regardless of whether the regressor contributes. However, adjusted $R^2$ only increeases if the additional regressor reduces $SS_{res}/(n-p) = MS_{res}$

---

### Test on the individual Regression Coefficients

Once we have test at least one of regressor is important. We must decide whether the increase in the regression sum of squares is suffiicient to warrant using the additional regressor in the model. adding an unimportant regressor may increase the $MS_{res}$ instead of decrease the usefulness of the model. 

We test $H_0 :\beta_k = 0$ against $H_1: \beta_k \ne 0$ 

If not rejected, then this indicates that the regressir can be deleted from the model. the test statistic is :

$$
t_0 = \frac{\hat{\beta_j}}{\sqrt{\sigma^2 C_{jj}}} = \frac{\hat{\beta_j}}{se(\hat{\beta_j})} \sim t(n-p)
$$

---

### Extra sum of square method

We use this method to determine contribution to regression. Recall we start from $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... \beta_kx_k + \epsilon = X\beta + \epsilon$

We shall now partition regressors into 2 sets : 

$$
y = X\beta + \epsilon = \begin{pmatrix} X_1 X_2\end{pmatrix} \begin{pmatrix} \beta_1 \\ \beta_2\end{pmatrix} + \epsilon = X_1 \beta_1 + X_2 \beta_2 + \epsilon
$$

where $\beta_1$ is $(p-r) \times 1$ and $\beta_2$ is $r \times 1$

this is the same as 

$$y = (\beta_0 + \beta_1x_1 + \dotsc + \beta_{k-r}x_{k-r}) +  (\beta_{k-r+1}x_{k-r+1} + \dotsc + \beta_{k}x_{k}) + \epsilon$$

Recall that $SS_R(\beta) \equiv \hat{\beta}X'y$, then $SS_R(\beta_2) = \hat{\beta_1}X_1'y$ (p-r degree of freedom)

We define $SS_R(\beta_2|\beta_1)$ as the extra sum of squares due to $\beta_2$ , it measures the increase in the regression sum of squares that results from adding the regressors $x_{k-r+1},\cdots,x_k$ to a model already containing $x_1,\dotsc,x_{k-r}$

$$
\begin{aligned}
    SS_r(\beta_2|\beta_1) &= SS_R(\beta) - SS_R(\beta_1) \\
    &= SS_R(\beta_1,\beta_2) - SS_R(\beta_1) \\
    &= SS_R(\beta_0 ...\beta_k) - 
    \underbrace{SS_R(\beta_0 ... \beta_{k-r})}_{\text{first p - r regressors}}
\end{aligned}
$$

In fact, we can find in general $SS_r(\beta_j|\beta_0,\cdots,\beta_{j-1},\beta_{j+1},\cdots\beta_k,)$. We can think of this as measuring the **contribution of $x_j$ as if it were the last variable added to the model

Consider only $SS_R(\beta_0)$, in this model $y_i = \beta_0 + \epsilon_i \implies y=X\beta_0 + \epsilon$. From above, 

$$
\begin{aligned}
    \hat{\beta_0} &=(X'X)^{-1}X'y = n^{-1}\sum y_i = \bar{y} \\
    &\implies SS_R(\beta_0) = \hat{\beta_0}X'y = \bar{y} \sum y_i = n\bar{y}^2
\end{aligned}
$$

Recall  our regressor sum of square : $SS_R = \hat{\beta}X'y - n\bar{y}^2$

$$
\begin{aligned}
    SS_R &= \hat{\beta}X'y - n\bar{y}^2 \\
    &= SS_R(\beta) - SS_R(\beta_0) \\
    &= SS_R(\beta_0 ...\beta_k) - SS_R(\beta_0) \\
    &= SS_R(\beta_1,\cdots,\beta_k|\beta_0)\\
    \implies SS_R &\text{ is the regression sum square due to } \beta_1,\cdots,\beta_k \text{ with } \beta_0 \text{ already in model}
\end{aligned}
$$

In fact, $SS_R$ is the sum of all the extra sum of squares 

$$
\begin{aligned}
    SS_R &= SS_R(\beta_0 ...\beta_k) - SS_R(\beta_0) \\
    &= SS_R(\beta_0,\beta_1) - SS_R(\beta_0) + (SS_R(\beta_0,\beta_1,\beta_2) - SS_R(\beta_0,\beta_1)) +\dotsc + SS_R(\beta_0 ...\beta_k) \\
    &= SS_R(\beta_1|\beta_0) + SS_R(\beta_2|\beta_0,\beta_1) + \cdots + SS_R(\beta_k|\beta_0,\dotsc,\beta_{k-1}) \\
\end{aligned}
$$

Note that $SS_R(\beta_0,\beta_1,\beta_2|\beta_0) \ne SS_R(\beta_1|\beta_2,\beta_0) + SS_R(\beta_2|\beta_1,\beta_0)$

----

Let FM denote the full model and RM denote the reduced model. To test the significance of $\beta_2$. we have the test statistic

$$
F = \frac{SS_R(\beta_2|\beta_1)/ r}{SS_{res}(FM)/n - p} = \frac{SS_R(\beta_2|\beta_1)/ r}{MS_{res}}
$$

WE can write $SS_R(\beta_2)$ in terms of $SS_{res}$ 

$$
\begin{aligned}
    SS_R(\beta_2|\beta_1) &= SS_R(\beta_2,\beta_1) - SS_R(\beta_1) \\
    &= SS_R(\beta_2,\beta_1) - SS_R(\beta_0) - [(SS_R(\beta_1)) - SS_R(\beta_0)] \\
    &= SS_T - SS_res(FM) - (SS_T -SS_{res}(RM)) \\
    &=SS_{res}(RM) - SS_res(FM) \\
    &\implies \text{degree of freedom of }SS_R(\beta_2|\beta_1) = \text{DF of } SS_{res}(RM) - \text{DF of } SS_{res}(FM) \\
     &\quad = [n - (p-r)] - (n-p) \\
     &\quad = r
\end{aligned}
$$

To obtain extra sum of squares, df or p-value of test we can use the following r-code

E.g.

1) We want to test $H_0: \beta_4 = \beta_5 = 0$. we call

```r
anova(lm(y~x1 + x2 + x3), lm(y~x1 + x2 + x3 + x4 + x5))
```

This way we can get $SS_R(\beta_4,\beta_5|\beta_0,\cdots,\beta_3) = SS_{res}(RM) - SS_{res}(FM)$

2) We want to test $H_0: \beta_1 = \beta_2 = \beta_3= 0$

```r
anova(lm(y~x4 + x5), lm(y~x1 + x2 + x3 + x4 + x5))
```

this will give us $SS_R(\beta_1,\cdots,\beta_3|\beta_4,\beta_5,\beta_0)$

Alternatively, if we call

```r
anova(lm(y~1), lm(y~x1 + x2 + x3))
```

we can get $SS_R(\beta_1,\cdots,\beta_3|\beta_0)$ and note that $F = \frac{MS_R(\beta_1,\cdots,\beta_3|\beta_0)}{MS_{res}(\beta_1,\beta_2,\beta_3,\beta_0)}$

3) Suppose we want $SS_R(\beta_2|\beta_0 , \beta_1)$ then we call

```r
anova(lm(y~x1), lm(y~x1 + x2))
```

if we just do 

```r
anova( lm(y~x1 + x2))
```

this will give us both : $SS_R(\beta_1| \beta_0)$ and $SS_R(\beta_2| \beta_1,\beta_0)$

## Testing General Linear Hypothesis

Suppose we want to test $T\beta = 0$ where $T$ is a $r \times p$ matrix where all the r equatiions in $T\beta = 0$ are indepedent

eg: $H_0 : \beta_1 = \beta_3, \beta_4 = 0, \beta_5 = 0$. then we have

$$
T = 
\begin{pmatrix}
    0 &1 &0 &-1 &0 &0 \\
    0 &0 &0 &0 &1 &0 \\
    0 &0 &0 &0 &0 &1\\
\end{pmatrix} \quad 
\beta = 
\begin{pmatrix}
    \beta_0 \\ \vdots \\ \beta_5
\end{pmatrix} \quad\implies
T\beta = 
\begin{pmatrix}
    \beta_1 - \beta_3 \\ \beta_4 \\ \beta_5
\end{pmatrix} = 
\begin{pmatrix}
    0 \\ 0 \\ 0
\end{pmatrix}
$$

Applying $H_0$ to the FM, we get a RM : $y_{n\times 1} = z_{n \times (p-r)} \Gamma_{(p-r) \times 1} + \epsilon_{n \times 1}$ eg

$$
\begin{aligned}
    y &= \beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_3x_3 + \beta_4x_4 + \beta_5x_5 \\
\text{given } H_0 \text{ is true,} y &= \beta_0 + \beta_1(x_1 + x_3) + \beta_2 x_2  + \epsilon \\
&= \gamma_0 + \gamma_0 z_1 + \gamma_2 z_ + \epsilon
\end{aligned}
$$

Similarly, the regression sum of square - denote as $SS_H$ is : $SS_H = SS_{res}(RM) - SS_{res}(FM)$ and , $DF(SS_H) = r$

Also $F = \frac{SS_H/ r}{SS_{res}(FM)/n - p}$

## Estimation of mean response = $E(y|x_0)$

- we know $E(y|x_0) = E(y_0) = \beta_0 + \beta_1 x_{01} + \cdots + \beta_k x_{0k} = x_0'\beta$ where $x_0 = \begin{pmatrix} 1 &x_{0k} &\cdots &x_{0k} \end{pmatrix}$
- and that the fitted value at $x_0'$ is $\hat{y_0} = x_0' \hat{\beta}$
- We can estimate $E(y|x_0)$ with $\hat{y_0}$, $E(\hat{y_0}) = E(x_0' \hat{\beta}) = x_0' E(\hat{\beta}) = x_0' \beta = E(y|x_0)$
- $Var(\hat{y_0}) = Var(x_0' \hat{\beta}) = x_0' Var(\hat{\beta}) x_0 = \sigma^2 x_0' (X'X)^{-1}x_0$
- We estimate the variance with $\hat{Var(\hat{y_0})} = \hat{\sigma^2}x_0' (X'X)^{-1}x_0$
- $\frac{\hat{y_0} - E(y|x_0)}{\sqrt{Var(\hat{y_0})}} \sim t(n-p)$
- $100(1-\alpha)$% CI for $E(y|x_0)$ is $\hat{y_0} \pm t_{\alpha /2}(n-p)\sqrt{\hat{\sigma^2}x_0' (X'X)^{-1}x_0}$
- we call in r 

```r
predict.lm(fitted.model, newdata = dataframe(...), interval = 'confidence', level = 0.95)
```
## Prediction of New observation $y^*_0$

- denote $y^*_0$ as the future response given some $x_0$
- we estimate $\hat{y^*_0} = x_0 \hat{\beta}$
- Mean of $y^*_0 - \hat{y^*_0} = 0$ since $E(y^*_0 ) = E(\hat{y^*_0}) = x_0\beta$
- Variance of $y^*_0 - \hat{y^*_0}$ is $Var(y^*_0 - \hat{y^*_0}) = Var(y^*_0) + Var(\hat{y^*_0}) - 2Cov(y^*_0,\hat{y^*_0}) = \sigma^2 + \sigma^2 x_0' (X'X)^{-1}x_0 = \sigma^2 (1+x_0' (X'X)^{-1}x_0)$
- $Cov(y^*_0,\hat{y^*_0})$ is 0 as the future response is indepedent of the data used to predict $y_0^*$
- of course, the estimated variance $\hat{Var(y^*_0 - \hat{y^*_0})} = \hat{\sigma^2}(1+x_0' (X'X)^{-1}x_0)$
- Similarly, $\frac{y^*_0 - \hat{y^*_0}}{\sqrt{Var(y^*_0 - \hat{y^*_0})}} \sim t(n-p)$
- And, Prediction interval (PI) is $\hat{y^*_0}  \pm t_{\alpha /2}(n-p)\sqrt{\hat{\sigma^2}(1+x_0' (X'X)^{-1}x_0)}$

## Simulatenous Confidence Intervals on Regression Coefficients

A set of confidence or prediction interval that is true simultaenously  with probability $1-\alpha$ are called simulatenous or joint confidence/predicition intervakl

We define a joint confidence region for the multiple regression model. (proof not shown) We may show that 

$$
\frac{(\hat{\beta} - \beta)'X'X(\hat{\beta} - \beta) /p}{MS_{res}} \sim F_{p,n-p}
$$

Construction of this joint interval is relatiely straightforward for p = 2 (simple linear regression) $\frac{(\hat{\beta} - \beta)'X'X(\hat{\beta} - \beta) /2}{MS_{res}} < F_{\alpha,2,n-2}$

$$
\frac{
\begin{pmatrix}
    \hat{\beta_0} - \beta_0 &\hat{\beta_1} - \beta_1
\end{pmatrix}
\begin{pmatrix}
    n & \sum x_i \\ \sum x_i & \sum x_i^2
\end{pmatrix}
\begin{pmatrix}
    \hat{\beta_0} - \beta_0  \\ \hat{\beta_1} - \beta_1
\end{pmatrix} /2
}{MS_{res}} =
\frac{n(\hat{\beta_0} - \beta_0)^2 + 2(\hat{\beta_0} - \beta_0)(\hat{\beta_1} - \beta_1)\sum x_i + (\hat{\beta_1} - \beta_1)^2 \sum x_i^2}{2 MS_{res}}< F_{\alpha,2,n-2}
$$

![Figure 3.8 from Douglas M et al](../images/Screenshot%202024-07-17%20Fig3.8%20D%20Montgomery.png)

this joint confidence regionis an ellipsoidal region. The tilt of the ellipse is a function of the covariance between $\hat{\beta_0}$ and $\hat{\beta_1}$ 
- A positive covariance implies that errors in the point estimates of beta are likely to be in the same direction, while a negative covariance indicates that these errors are likely to be in opposite direction.
- The elongation of the region depends on the relative sizes of the variances of $\beta_0$ and $\beta_1$  Generally, if the ellipse is elongated in the $\beta_0$ direction, this implies that $\beta_0$ is not estimated as precisely as $\beta_1$

### Alternative approach: Bonferroni CI

We now construct a $(1 - \frac{\alpha}{m}) 100$% CI for each parameter. this can provide us a joint coverage probability of $\ge 1 - \alpha$

this is called a bonferroni confidence region - which is a rectangular region. uses principle of bonferroni inequality

For p intervals, $\hat{\beta_j} \pm t_{\alpha /2p} (n-p) \sqrt{\hat{\sigma^2}C_{jj}}$

## hidden extrapolation in multiple regression

In predicting new responses and in estimating mean response with a given point $x_0$, one must be careful about extrapolating beyond the region of the original observations

![Figure 3.10 from Douglas M et al](../images/2024-07-17%20Fig3.10%20D%20Montgomery.png)

the point on $x_{01}$ and $x_{02}$ is beyond range of experiment, even though it lies in range of both regressor. 

we define the smallest convex set containing all original data point as the regressor variable hull (RVH)
- if $x_0 \in RVH \implies$ interpolation
- if $x_0 \notin RVH \implies$ extrapolation

In fact the hat values (diagonals $h_{ii}$ of the hat matrix - $H = X(X'X)^{-1}X'$) are useful in detecting hidden extrapolation. 
- $h_{ii}$ depends on both the euclidean distance of the point $x_0$ from the centriod and the density of points in the RVH. 
- In general the point with largest $h_{ii}$ or $h_{max}$ will lie on the boundary of RVH
- $\{ x|X(X'X)^{-1}X' \lt h_{max}\}$ represents an ellipsoid containing all points in the RVH
- if we are interested in predicting or estimating at point $x_0$, the location of that point relative to RVH is reflected by 
  
$$
h_{00} = x_0(X'X)^{-1}x_0'
$$
- if $h_{00} > h_{max} \implies x_0$ is outside ellipsoid
- if $h_{00} \le h_{max} \implies x_0$ is inside ellipsoid but could be inside RVH

to get the hat values, we call in r

```r
H = X %*% XPXI %*% t(X)
Hii = diag(H)
```

to get $h_{00}$, we call in r

```r
x0 = array(x(1,20,250),dim =3)
H00 = t(x0) %*% XPXI %*% x0
```

Note that $x0$ in the code is a column vector

## Standardised regression coefficient

We compare regression coefficients by using standardized regression coefficients.

### Unit Normal Scaling (standardised normal)

$$
z_{ik} = \frac{x_{ij} - \bar{x_j}}{s_j}, \text{where } s^2_j = \frac{\sum (x_{ij} - \bar{x_j})^2}{n-1} \quad \text{ is the sample variance of regressor}
$$

and 

$$
y^*_i = \frac{y_i - \bar{y}}{\sqrt{\frac{1}{n-1}\sum (y_i - \bar{y})^2}}
$$

All the scaled regressors and scaled responses have sample mean = 0 and sample variance equal to 1

Using the new variables, the regression model becomes $y^*_i = b_1 z_{i1} + \cdots + b_kz_{ik} + \epsilon_i$ and $Z^TZ \hat{b} = Z^T y$

the least squares estimator of b is then : $\hat{b} = (Z^TZ )^{-1}Z^T y^*$

Note: there is no intercept in this model

### Unit Length Scaling

we set : 

$$
w_{ik} = \frac{x_{ij} - \bar{x_j}}{s_{jj}^{1/2}}, \text{where } s_{jj} = \sum (x_{ij} - \bar{x_j})^2 \quad \text{ is the corrected sum of squares for regressor }
$$

and 

$$
y^0_i = \frac{y_i - \bar{y}}{\sqrt{\sum (y_i - \bar{y})^2}}
$$

Each new regressor has mean = 0 and length $\sqrt{\sum (w_{ij} - \bar{w_i})^2} = 1$

- sum of $w_{ik} = 0$

the regression model is $y^0_i = b_1 w_{i1} + \cdots + b_kw_{ik} + \epsilon_i$

similarly, the least squares regression coefficient is $\hat{b} = (W^TW )^{-1}W^T y^0$

In fact for the unit length scaled variables, the off diagonals of $W^TW$ are the correlation coefficient of regression variables. $r_{ij}$ is the correlation coefficient between $x_i$ and $x_j$. **This is known as correlation matrix**

$$
W^TW = \begin{pmatrix}
    1 &r_{12} &\cdots &r_{1k} \\
    r_{21} &1 & \cdots &r_{2k} \\
    \vdots &\vdots & &\vdots \\
    r_{k1} &r_{k2} &\cdots &1
\end{pmatrix}
$$

Recall that 

$$
r_{ij} = \frac{S_{ij}}{\sqrt{S_{ii}S_{jj}}} = \frac{\sum_k (x_{kj} - \bar{x_i})(x_{kj} - \bar{x_j})}{\sqrt{\sum (x_{ik} - \bar{x_i})^2 \sum (x_{jk} - \bar{x_j})^2}}
$$

and 

$$
W^T y^0 = \begin{pmatrix}
    r_{1y} \\ r_{2y} \\ \vdots \\r_{ky}
\end{pmatrix} \quad \text{where } r_{jy} \text{ is the simple correlation between regressor and response}
$$

<details>

<summary>proof</summary>

$$
\text{the first entry of }W^T y^0 \implies w_{11}y_1 + \cdots + w_{n1}y_n = \sum_k \frac{(x_{k1} - \bar{x_1})(y_k - \bar{y})}{\sqrt{\sum (x_{k1} - \bar{x_1})^2 \sum (y_k - \bar{y})^2}} = r_{1 y}
$$

from here,

$$
r_{jy} = \frac{S_{jy}}{\sqrt{SS_T S_{jj}}} = \frac{\sum_k (x_{kj} - \bar{x_i})(y_k - \bar{y})}{\sqrt{\sum (x_{ik} - \bar{x_i})^2 \sum (y_k - \bar{y})^2}}
$$

</details>

Note that : $Z^TZ = (n-1)W^TW \iff \frac{1}{n-1}Z^TZ = W^TW$

The rationale for scaling :
- regression coefficients can be compared meaningfully because of they now have a common unit of measurement
- reduce effect of multicollinearity(next part)
- response variable is caled so that resulting model has intercept term that is zero. 

<details>

<summary>proof that intercept is 0</summary>

we set $S = \sum(y_i^0 - \hat{y_i^0})^2$

$$ 
\begin{aligned}
    \frac{\partial S}{\partial b_0} &= \sum 2 (y_i^0 - (b_0 + b_1 w_{i1} + \cdots + b_kw_{ik})) \quad \text{set to } 0 \\
    \sum y_i^0  &= \sum (b_0 + b_1 w_{i1} + \cdots + b_kw_{ik}) \\
    \sum y_i^0 &= nb_0 + b_1 \sum w_{i1} + \cdots + b_k \sum w_{ik}\\
    \because \text{sum of } w_{ik} = 0, &\implies \hat{b_0} = 0
\end{aligned}
$$

</details>

## Multicollinearity (brief discussion)

Multicollinearity is the near linear-dependence among the regressor variables. An exact linear depedence leads to a singular $X^TX$ The presence of near-linear dependence would result in the inability to estimate regression coefficients accurately. 

### VIFs - variance inflation factors as a diagnostic for multicollinearity 

Suppose we use the unit length scaling so that $X^TX = W^TW$. this wiy we get the form of a **correlation matrix**

E.g. : $W^TW = \begin{pmatrix} 1 &0.824 \\ 0.824 &1 \end{pmatrix}$ and $(W^TW)^{-1} = \begin{pmatrix} 3.11841 &-2.57 \\ -2.57 &3.11841\end{pmatrix}$

this implies $Var(\hat{b_1}) = Var(\hat{b_2}) = 3.11841 \sigma^2$. Here variances are inflated due to multicollinearity, this is evident from the non-zero off-diagonal elements in $W^TW$ - which the correlation between the regressor.



Main diagonals of $(X^TX)^{-1}$ are called **variance inflation factors (VIFs)**

From above example, $VIF_1 = VIF_2 = 3.11841$
- if VIF =1 , then the regressors are orthogonal
- in fact, we can show that $VIF_j = \frac{1}{1-R^2_j}$ where $R^2_j$ is the coefficient of multiple determination obtained from regressing $X_j$ against other regression coefficient

If $X_j$ is nearly dependent on some subset of remaining regressor variables. $R_j^2 \rightarrow 1$ and $VIF_j >> 1$. if orthogonal, then  $R_j^2 \rightarrow 0$ and $VIF_j \rightarrow 1$

Essentially, VIFs measure how much variance of regression coefficient $B_j$ is affected by its relationship of $X_J$ and other regressors.

Threshold:
- VIF > 2.5 (implies some evidence of multicollinearity)

in R:

```r
Vif(fitted.model)
```

---

mutlicollinearity inflates the variances of coefficients and increase probability that the coefficients will have the wrong sign.

Other reasons why a wrong sign may appear include:
- small range of regressor. if X to close together, variance of $\beta$ will be large. $Var(\beta_1) = \sigma^2 / \sum (x_i1 - \bar{x1})^2$. Small spread in x, lead to larger variance. 
- Not including other important regressors
- Wrong data used