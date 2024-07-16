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

It is more convenient to deal with this using matrix notation: $\bold{y = X \beta + \epsilon}$

$$
\begin{aligned}
&\bold{y} =
\begin{pmatrix}
y_1\\y_2\\ \vdots \\ y_n
\end{pmatrix}, \quad && \bold{X} = 
\begin{pmatrix}
1 & x_{11} &\cdots & x_{1k}\\
1 & x_{21} &\cdots & x_{2k}\\
\vdots & \vdots & &\vdots\\
1 & x_{n1} &\cdots & x_{nk}
\end{pmatrix} \\
&\bold{\beta} = 
\begin{pmatrix}
\beta_1\\ \beta_2\\ \vdots \\ \beta_k
\end{pmatrix}, \quad &&\bold{\epsilon} = 
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

We now proceed to find the vector of least squares estimatators $\bold{\hat{\beta}}$

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

If we let $C=(X'X)^{-1}$, then $Var(\hat{\beta}_j) = \sigma^2 C_{jj}$ and $Cov(\hat{\beta}_i,\hat{\beta}_j) = = \sigma^2 C_{ij}$

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


