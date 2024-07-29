Notes taken from **Douglas C M et al, Introduction to Linear Regression Analysis**

# Polynomial models

Considerations when it comes to fitting a polynomial in one variable. 

1) Order of model. It is important to keep the model as low as possible. When response appears to be curvilinear, transformations should be tried to keep the model first order. avoid higher order polynomials k>2
2) Model buidling strategy (discussed in Variable Selection)
3) Dont extrapolate beyond original data
4) Ill conditioning. as the order of polynomial increases the $X^TX$ matrix becomes ill conditioned. ie becomes nearly singular. We can attempt to reduce thus by centering the data about its means ie use $(X_i - \bar{X})$ as regressor varibale instead. 
   - if values of x are limted to narrow range eg [1,2] then there can be significant ill-conditioning or multicollinearity in the columns of X, esp if there is a quadratic term $X^2$
5) Hierarchy - Hierarchical models contain all terms of order k and lower. if missing any single power, it is a non-hierarchical model.  hierarchical models are invariant under linear transformation of regressor variable. 

The best advice is to fit a model that has all terms significant and to use discipline knowledge rather than an arbitrary rule as an additional guide in model formulation.

## Splines - Piece-wise Polynomials

Here we divide the range of x into different segments and fit a polynomial (spline) in each segment
- splines are piecewise polynomila of order k
- the joint points of the segments are called knots. 

We do splines fitting when a low order polynomial provides a poor fit to data, and increasing the order of the polynomial does not improve the fit substantially. 

### cubic spline is a polynomial of order 3

$y = \beta_0 + \cdots + \beta_3 x^3 + \epsilon$

A cubic spline with h knots and continuous at knots is given by:

$$
E(y) = S(x) = \sum_j^3 \beta_{0j}x^j + \sum_i^h \beta_i (x-t_i)^3_+ \quad \text{where }(x-t_i)_+ = \begin{cases}
    (x-t_i) \quad \text{if } x-t_i \gt 0 \\
    0 \quad \text{if } x-t_i \le 0 \\
\end{cases}
$$

For Example: we consider case of2 knots (h =2):

$$
\begin{aligned}
    E(y) = S(X) &= \sum_{j=0}^3 \beta_{0j}x^j + \sum_{i=1}^2 \beta_i (x-t_i)^3_+ \\
    &= \beta_{00} + \beta_{01}x + \beta_{02}x^2 + \beta_{03}x^3 + \beta_1(x-t_1)^3_+ + \beta_2(x-t_2)^3_+
\end{aligned}
$$

- if $x \lt t_1$, $S(X) = \beta_{00} + \beta_{01}x + \beta_{02}x^2 + \beta_{03}x^3$
- $t_1 \lt x \lt t_2$, $S(X) = \beta_{00} + \beta_{01}x + \beta_{02}x^2 + \beta_{03}x^3 + \beta_1(x-t_1)^3$ 
- if $x \gt t_2$, $S(X) = \beta_{00} + \beta_{01}x + \beta_{02}x^2 + \beta_{03}x^3 + \beta_1(x-t_1)^3 + \beta_2(x-t_2)^3$

to get the coefficients of the spline, we can fit a multiple linear regression model to get the individual betas.

eg $= \beta_{00} + \beta_{01}x + \beta_{02}x^2 + \beta_{03}x^3 + \beta_1(x-6.5)^3_+ + \beta_2(x-13)^3_+$

Modify the data set such that for values <6.5 and <13 , we set the values = 0. in r

```r
data.spline = data.frame(y,x1,x2,x3,x4,x5)
model.spline = lm(y ~ x1+x2+x3+x4+x5)

#plot
fitted = predict(model.spline)
plot(x,y,pch=15,col="red")
par(new=TRUE)
plot(x,fitted,axes=FALSE,type ='l', col = "black")

#hypothesis test
anova(model.spline)
anova(lm(y~1),model.spline)
```

### No continuity restriction on Spline

Suppose now we do not require our splines to be continuous, then we have :

$$
E(y) = S(x) = \sum_{j=0}^3 \beta_{0j}x^j + \sum_{i=1}^h \sum_{j=0}^3\beta_{ij} (x-t_i)^j_+ \quad \text{where }(x-t_i)_+^0 = 1 \text{ if } x>t \text{ and } 0 \text{ otherwise}
$$

- the term$ \beta_{i0} (x-t_i)^0$ in the model forces a discontinuity at t.
- if the term is absent S(X) is continuous at t
- the fewer continuity restrictions required, the better is the fit since more paramters are in the model. 

Consider the case of cubic spline with one knot - i.e. h =1

$$
E(y) = S(x) = \beta_{00} + \beta_{01}x + \beta_{02}x^2 + \beta_{03}x^3 + \beta_{10}(x-t_1)^0_+ + \beta_{11}(x-t_1)^1_+ + \beta_{12}(x-t_1)^2_+ + \beta_{13}(x-t_1)^3_+
$$

- for $x \le t_1, S(x) = \beta_{00} + \beta_{01}x + \beta_{02}x^2 + \beta_{03}x^3$
- for $x \gt t_1, S(x) = \beta_{00} + \beta_{01}x + \beta_{02}x^2 + \beta_{03}x^3 + \beta_{10} + \beta_{11}(x-t_1)^1 + \beta_{12}(x-t_1)^2 + \beta_{13}(x-t_1)^3$

if we want a linear spline with h knots and no continuity:

$$
E(y) = S(x) = \sum_{j=0}^1 \beta_{0j}x^j + \sum_{i=1}^h \sum_{j=0}^1\beta_{ij} (x-t_i)^j_+
$$

if we want linear spline w h knots and continuous:

$$
E(y) = S(x) =\sum_{j=0}^1 \beta_{0j}x^j + \sum_{i=1}^h \beta_i (x-t_i)^1_+
$$

# Non Parametric Models

We develop a model -free basis for predicting the response over the range of data

Consider, ordinary least squares and recall that $\hat{y} = X\hat{\beta} = X(X'X)^{-1}X'y = Hy = \begin{pmatrix}
    h_{11} & h_{12} &\cdots &h_{1n} \\
    h_{21} & h_{22} &\cdots &h_{2n} \\
    \vdots & & &\vdots \\
    h_{n1} &h_{2n} &\cdots &h_{nn}
\end{pmatrix} \begin{pmatrix} y_1 \\y_2\\ \vdots \\ y_n \end{pmatrix}$

As a result, $\hat{y_i} = \sum_j h_{ij} y_j$ The predicted value for ith response is simply a linear combination of the original data. (this concept will be used subsequently)

## Kernel Regression

one of the alternative non-parametric approaches is the kernel smoother. It uses the weighted average of the data. 

for a kernel smoother:

$$
\tilde{y_i} = \sum_j  w_{ij}y_j
$$

above, sum of weights = 1 And  we have $\tilde{y}=Sy$ where S is the smoothing matrix. 

The weights are chosen such that $w_{ij} = 0$ for all $y_i$ outside the a defined 'neighbourhood" of the specific location of interest.

Kernal Smoothers use a bandwidth, b to define this neighbourhood of interest. 
- a large value of b results in more of the data being used to predict the response at the specific location.
- As b decreases, less of the data are used to generate the prediction, this causes resulting plot to look more wiggly. 

Typically, the kernel functions hvae the following properties:
  - $K(t) \ge 0$ for all t
  - $\int_{-\infin}^{\infin} K(t) dt =1$
  - $K(-t) = K(t)$ symmetry

these are also properties of a symmetric probability density function.

The specific weights for the kernel smoother are given by:

$$
w_{ij} = \frac{K(\frac{x_i - x_j}{b})}{\sum_{k=1} K(\frac{x_i - x_k}{b})}
$$

![Table 7.5 from D Montgomery et al](../images/Screenshot%202024-07-25%20Table7.5%20D%20Montgomery.png)

eg for box function, 

$$
K(\frac{x-x_k}{b}) =1 \iff x-0.5b \le x_k \le x+ 0.5b
$$

weight is heavier if given point x is closer to $x_k$

Then, we find $w_{j} = \frac{K(\frac{x - x_j}{b})}{\sum_{k=1} K(\frac{x - x_k}{b})}$ and subsequently get $\tilde{y} = \sum_j  w_{j}y_j$ for the given x. 

example of r code - normal K
```r
library(np)
npmodel = npreg(y~x,ckertype = "gaussian",bws = 2)
```
## LOESS - locally weighted regression

Like kernel regression, Loess uses the data from a neighbourhood around the specific location. The neighbourhood is defined as the span, which is the fraction of total points used to form neighbourhoods. A span of 0.5 indicdates that the closest half of the total data points used as the neighbourhood. 

The loess procedure then uses the points in the neighbourhood to generate a weighted least squares estimate of the specific response. 

The weights for the weighted least squares portion of the estimation are based on the distance of the points used in the estimation from the specific location of interest.
- this involves finding the nearest neigbours using euclidean distance

most software packages use the **tri-cube** weighting function. 

We let $x_0$ be the specific location of interest. and let $\Delta(x_0)$ be the distance the farthest point in neighbourhood lies from the specific location of interest

the tri-cube weight function is given by

$$
W[\frac{|x_0 - x_j|}{\Delta(x_0)}]
$$

where

$$
W(t) = 
\begin{cases}
    (1-t^3)^3 \text{, if } 0 \le t \lt 1 \\
    0 \quad \text{, otherwise}
\end{cases}
$$

in r 

```r
library(np)
loess.model = loess(y~x)
```

### Estimating $\sigma^2$ on the loess model

We have $\tilde{y}=Sy$, where S is the smoothing matrix of the loess model

$$
\begin{aligned}
    SS_{res} &= \sum_i (y_i - \tilde{y_i})^2\\
    &= (y-Sy)'(y-Sy) \\
    &= y'(I-S')(I-S)y \\
    &= y'(I - S' -S + S'S)y
\end{aligned}
$$

The asymptotic expected value for $SS_{res}$ :

$$
\begin{aligned}
    Trace[(I - S' -S + S'S)\sigma^2 I] &= \sigma^2 [trace(I) - trace(S') - trace(S) + trace(S'S)] \\
    &= \sigma^2(n-2trace(S) + trace(S'S))
\end{aligned}
$$

we have a common estimate of $\sigma^2$

$$
\tilde{\sigma^2} = \frac{\sum_i (y_i - \tilde{y_i})^2}{n-2trace(S) + trace(S'S)}
$$

### Other notes:
from **Gareth J et al., An introduction to Statistical Learning**