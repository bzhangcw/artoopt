---
bibliography: [../ref.bib]
title: QAP
---

# QAP, the problem

QAP, and alternative descriptions, see @jiang_l_p-norm_2016

$$\begin{aligned}
&\min_X f(X) = \textrm{tr}(A^\top XB X^\top)  \\
& = \textrm{tr}(X^\top A^\top XB) & x = \textrm{vec}(X)\\
& = \left <\textrm{vec}(X),  \textrm{vec}(A^\top X B )  \right > \\
& = \left <\textrm{vec}(X), B^\top \otimes A^\top \cdot \textrm{vec}(X)  \right > \\ 
& = x^\top (B^\top \otimes A^\top) x\\ 
\mathbf{s.t.} & \\ 
&X \in \Pi_{n}
\end{aligned}$$

is the optimization problem on permutation matrices:

$$ \Pi_{n}=\left\{X \in \mathbb R ^{n \times n} \mid X e =X^{\top} e = e , X_{i j} \in\{0,1\}\right\}$$

The convex hull of permutation matrices, the Birkhoﬀ polytope, is defined:

$$D _{n}=\left\{X \in \mathbb R ^{n \times n} \mid X e =X^{\top} e = e , X \geq 0 \right\}$$

for the constraints, also equivalently:
$$\begin{aligned}
& \textrm{tr}(XX^\top) = \left <x, x \right >_F= n, X \in D_{n}
\end{aligned}$$

## Differentiation

$$\begin{aligned}
&  \nabla f = A^\top XB + AXB^\top \\
& \nabla \textrm{tr}(XX^\top) = 2X
\end{aligned}$$

# $\mathscr L_p$ regularization

various form of regularized problem:

- $\mathscr L_0$: $f(X) + \sigma ||X||_0$ is exact to the original problem for efficiently large $\sigma$ @jiang_l_p-norm_2016, but the problem itself is still NP-hard.
  
- $\mathscr L_p$: also suggested by @jiang_l_p-norm_2016, good in the sense:
  - strongly concave and the global optimizer must be at vertices
  - **local optimizer is a permutation matrix** if $\sigma, \epsilon$ satisfies some condition. Also, there is a lower bound for nonzero entries of the KKT points 

$$\min _{X \in D _{n}} F_{\sigma, p, \epsilon}(X):=f(X)+\sigma\|X+\epsilon 1 \|_{p}^{p}$$

- $\mathscr L_2$, and is based on the fact that $\Pi_n =  D_n  \bigcap \{X:\textrm{tr}(XX^\top) = n\}$, @xia_efficient_2010

$$\min_Xf(X)+\mu_{0} \cdot \textrm{tr} \left(X X^{\top}\right)$$

## $\mathscr L_2$

### naive 
$$\begin{aligned}
&\textrm{tr}(A^\top XB X^\top) + \mu_0 \cdot \textrm{tr}(X X^{\top}) \\
= & x^\top (B^\top \otimes A^\top + \mu\cdot  \mathbf e_{n\times n}) x\\ 
\end{aligned} $$
this implies a LD-like method. (but not exactly)

### exact penalty

$$\begin{aligned}
F_{\mu} & =  f  + \mu\cdot | \textrm{tr}(XX^T) -  n| \\
 &= f  + \mu\cdot n - \mu\cdot \textrm{tr}(XX^T)
\end{aligned}$$

very likely to become a concave function, cannot be directly solved by conic solver.

#### Derivatives

$$\begin{aligned}
\nabla F_\mu  = A^TXB + AXB^T - 2\mu X
\end{aligned}$$

Then, 

> projected gradient

by solving:

$$\begin{aligned}
&\min_D ||\nabla F_\mu - D ||_F^2  \\
\mathsf{s.t.} & \\
&D e = D^\top e = 0
\end{aligned}$$

then $\forall \alpha \gt 0 ,X + \alpha D \in D_n$

alternatively compute a new point then project onto $D_n$.

# Reference