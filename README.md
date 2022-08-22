# Polynomial-Approximator
A program to approximate continuous functions with polynomials on any given interval using least squares regression

Suppose we had some function $f(x)$ that we wanted to approximate using a polynomial. One obvious method is the taylor series of $f(x)$ centered on some $c$ but the error of the series is distributed unequally along the domain.  

Furthermore, taylor series for arbitrary functions may have intervals of convergance, such as $e^{-x^2}$ only converging on the interval $[-2, 2]$. This algorithm allows you to surpass this boundary. For example, the 10th taylor polynomial about $0$ is 

$$1-x^{2}+\frac{1}{2}x^{4}-\frac{1}{6}x^{6}+\frac{1}{24}x^{8}-\frac{1}{120}x^{10}$$

while the 10th polynomial approximation on the interval $[-3, 3]$ is about

$0.9910710040287763-0.9121511249107619x^{2}+0.3533078594251453x^{4}-0.06864072857794135x^{6}+0.0065184693000363x^{8}-0.00023962430100394564x^{10}$

Though the taylor polynomial outperforms this algorithm for values very close to $0$, the algorithm outperforms the taylor polynomial on the interval $[-2, 2]$ by a over a factor of $100$ when using the $L^2$ norm, and even extends this accuracy to the original interval of $[-3, 3]$

To derive this algorithm you first need to formalize the notion of error between the function $f(x)$ and the approximation $p(x)$. To do this we use the $L^2$ norm. the $L^2$ norm is the same as uclidean distance, the root of the sum of the squared differences in each dimention. 

$$E = \sqrt{\sum_{x=0}^{n}(f_x-p_x)^2}$$

For continuous functions, all we do is replace the sum with an integral, and change the bounds to the desired interval.

$$E=\sqrt{\int_{a}^{b}(f(x)-p(x))^2\mathrm{d} x} $$

Our goal will be tweaking the coefficients of $p(x)$ such that the error $E$ is minimized. Because the square root function is always increasing, minimizing the error is equivalent to minimizing the function 

$$E'=\int_{a}^{b}(f(x)-p(x))^2\mathrm{d} x$$

$p(x)$ can be expressed as a power series.

$$p(x) = \sum_{i = 0}^{n}c_ix^i$$

Since the objective of the algorithm is to minimize the error by changing $c_0,c_1...c_n...c_k$, p(x) is actually a function of all these variables.

$$p(c_0,c_1...c_i...c_n,x) = \sum_{i = 0}^{n}c_ix^i$$

The error function changes as well

$$E'(c_0,c_1...c_i...c_n)=\int_{a}^{b}(f(x)-p(c_0,c_1...c_i...c_n,x))^2\mathrm{d} x$$

Minimizing the error function is now simply just finding where the gradient of $E'$ is equal to $0$.

$$\nabla E'=0$$

Let's examine the partial derivatives one by one

$$\frac{\partial }{\partial c_i}\int_{a}^{b}(f(x)-p(c_0,c_1...c_i...c_n,x))^2\mathrm{d} x = 0$$

Using the chain rule,

$$\int_{a}^{b}(f(x)-p(c_0,c_1...c_i...c_n,x))\frac{\partial }{\partial c_i}p(c_0,c_1...c_i...c_n,x)\mathrm{d} x=0$$
$$\int_{a}^{b}f(x)\frac{\partial }{\partial c_i}p(c_0,c_1...c_i...c_n,x)\mathrm{d} x=\int_{a}^{b}p(c_0,c_1...c_i...c_n,x)\frac{\partial }{\partial c_i}p(c_0,c_1...c_i...c_n,x)\mathrm{d}x$$

since 
$$p(c_0,c_1...c_i...c_n,x) = \sum_{i = 0}^{n}c_ix^i$$
$$\frac{\partial }{\partial c_i}p(c_0,c_1...c_i...c_n,x)=x^i$$

Thus
$$\int_{a}^{b}p(c_0,c_1...c_i...c_n,x)x^i\mathrm{d}x=\int_{a}^{b}f(x)x^i\mathrm{d}x$$

The left side can be simplified further

$$\int_{a}^{b}p(c_0,c_1...c_i...c_n,x)x^i\mathrm{d}x$$

$$\int_{a}^{b}\sum_{j = 0}^{n}c_jx^j\cdot x^i\mathrm{d}x$$

$$\int_{a}^{b}\sum_{j = 0}^{n}c_jx^{i+j}\mathrm{d}x$$

$$\sum_{j = 0}^{n}c_j\int_{a}^{b}x^{i+j}\mathrm{d}x$$

$$\sum_{j = 0}^{n}\frac{c_j}{i+j+1}(b^{i+j+1}-a^{i+j+1})$$

finally, we get

$$\sum_{j = 0}^{n}\frac{b^{i+j+1}-a^{i+j+1}}{i+j+1}c_j=\int_{a}^{b}f(x)x^i\mathrm{d}x$$

For each $i$ from $0$ to $n$. Note that each coefficient $c_i$ is linear, thus there is only 1 solution that minimizes the error. The system can be written as a matrix.

$$K_{i+j+1} = \frac{b^{i+j+1}-a^{i+j+1}}{i+j+1}$$

$$K = \begin{bmatrix}K_{1}&\dots&K_{n+1}\\\ \vdots&\ddots&\vdots \\\ K_{n+1}&\dots&Kb_{2n+1}\end{bmatrix}$$
 
$$c=\begin{bmatrix} c_{0} \\\ \vdots \\\ c_{n} \end{bmatrix}$$
 
$$I = \begin{bmatrix} \int_{a}^{b}f(x)x^0 \mathrm{d}x \\\ \vdots \\\ \int_{a}^{b}f(x)x^n\mathrm{d}x\end{bmatrix}$$

$$Kc=I$$

Identifying the correct values for $c$ is simply inverting $K$ and multiplying by $I$

$$c = K^{-1}I$$

The $K$ matrix is initialized as a numpy array, along with the $I$ vector

```python
K = np.array([[(((b ** (i + j + 1)) - (a ** (i + j + 1)))/(i + j + 1)) for i in range(n)] for j in range(n)])
```

```python
I = np.array([quad(lambda x: f(x) * x ** i, a, b)[0] for i in range(n)])
```

The $c$ vector is calculated with a matrix multiplication, and `Polynomial()` presents the answer as a polynomial.