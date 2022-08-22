# Polynomial-Approximator
A program to approximate continuous functions with polynomials on any given interval using least squares regression

Suppose we had some function $f(x)$ that we wanted to approximate using a polynomial. One obvious method is the taylor series of $f(x)$ centered on some $c$ but the error of the series is distributed unequally along the domain.  

Furthermore, taylor series for arbitrary functions may have intervals of convergance, such as $e^{-x^2}$ only converging on the interval $[-2, 2]$. This algorithm allows you to surpass this boundary. For example, the 10th taylor polynomial about $0$ is 

$$1-x^{2}+\frac{1}{2}x^{4}-\frac{1}{6}x^{6}+\frac{1}{24}x^{8}-\frac{1}{120}x^{10}$$

while the 10th polynomial approximation on the interval $[-3, 3]$ is

$$0.9910710040287763-0.9121511249107619x^{2}+0.3533078594251453x^{4}-0.06864072857794135x^{6}+0.0065184693000363x^{8}-0.00023962430100394564x^{10}$$

Though the taylor polynomial outperforms this algorithm for values very close to $0$, the algorithm outperforms the taylor polynomial on the interval $[-2, 2]$ by a over a factor of $100$ when using the $L^2$ norm, and even extends this accuracy to the original interval of $[-3, 3]$

To derive this algorithm you first need to formalize the notion of error between the function $f(x)$ and the approximation $p(x)$. To do this we use the $L^2$ norm. the $L^2$ norm is the same as uclidean distance, the root of the sum of the squared differences in each dimention. 

$$E = \sqrt{\sum_{x=0}^{n}(f_x-p_x)^2}$$

For continuous functions, all we do is replace the sum with an integral, and change the bounds to the desired interval.

$$E=\sqrt{\int_{a}^{b}(f(x)-p(x))^2\mathrm{d} x} $$

Our goal will be tweaking the coefficients of $p(x)$ such that the error $E$ is minimized. Because the square root function is always increasing, minimizing the error is equivalent to minimizing the function 

$$E'=\int_{a}^{b}(f(x)-p(x))^2\mathrm{d} x$$

$p(x)$ can be expressed as a power series.

$$p(x) = \sum_{n = 0}^{k}c_nx^n$$

Since the objective of the algorithm is to minimize the error by changing $c_0,c_1...c_n...c_k$, p(x) is actually a function of all these variables.

$$p(c_0,c_1...c_n...c_k,x) = \sum_{n = 0}^{k}c_nx^n$$

