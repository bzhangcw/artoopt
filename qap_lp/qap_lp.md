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

## $\mathscr L_2$ + $\mathscr L_1$ penalized formulation 

Motivated by the formulation using trace:

$$\begin{aligned}
& \min_X  \textrm{tr}(A^\top XB X^\top) \\
\mathbf{s.t.} &\\
&   \textrm{tr}(XX^\top ) -  n = 0 \\
& X \in D_n
\end{aligned}$$

using absolute value of $\mathscr L_2$ penalty and by the factor that $\forall X \in D_n ,\; \textrm{tr}(XX^\top)\le n$, we have:

$$\begin{aligned}
F_{\mu} & =  f  + \mu\cdot | \textrm{tr}(XX^\top ) -  n| \\
 &= \textrm{tr}(A^\top XB X^\top)  + \mu\cdot n - \mu\cdot \textrm{tr}(XX^\top )
\end{aligned}$$

For sufficiently large penalty parameter $\mu$, the problem solves the original problem.

The penalty method is very likely to become a concave function (even if the original one is convex), and thus it cannot be directly solved by conic solver. 

### Projected gradient


```python
# code see
qap_lp.models.qap_model_l2.l2_exact_penalty_gradient_proj
```

Suppose we do projection on the penalized problem $F_\mu$ 

#### derivatives

$$\begin{aligned}
& \nabla_X F_\mu  = A^\top XB + AXB^\top - 2\mu X \\
& \nabla_\mu F_\mu  = n - \textrm{tr}(XX^\top) \\
& \nabla_\Lambda F_\mu  = - X
\end{aligned}$$

#### projected derivative

problem *PD*, a quadratic program

$$\begin{aligned}
&\min_D ||\nabla F_\mu + D ||_F^2  \\
\mathbf{s.t.} & \\
&D e = D^\top e = 0 \\ 
&D_{ij} \ge 0 \quad \textsf{if: } X_{ij} = 0\\
\end{aligned}$$

- There is no degeneracy, great.

### Remark

#### integrality of the solution
Computational results show that the *residue of the trace*: $|n - \textrm{tr}(XX^\top)|$ is almost zero, this means the algorithm converges to an integral solution. (even without any tuning of penalty parameter $\mu$)

Prove that it is exact if $\mu$ is sufficiently large, the model converges to an integral solution.

**PF. outline** the model uses exact penalty function, as $\mu$, actually $\{\mu_k\}$ become sufficiently large, the penalty method solves the original problem.



#### quality of the solution
it is however hard to find a global optimum, and gradient projection as defined above converges to a local integral solution and then stops, see instances with gap > 10%.


#### analytic representation for projection

in projected gradient method,
let the space of $D$, ($e$ is the vector of 1s)

 $$\mathcal D = \{D\in\mathbb{R}^{n\times n} : \; D e = D^\top e = 0;\; D_{ij} = 0,\;\forall  (i,j) \in M \}$$

is there a way to formulate the set for $F$ such that $\left <F, D \right>_F = 0, \; \forall D\in \mathcal D$, can we find an analytic representation?

#### what if projection is zero?


dual problem for $PD$

-  $\alpha,\beta,\Lambda$ are Lagrange multipliers, $\mathbf I$ is the identity matrix for active constraints of the $X \ge 0$  where $\mathbf I_{ij} = 1$ if $X_{ij} = 0$

$$\begin{aligned}
& L_d = 1/2\cdot ||\nabla F_\mu + D ||_F^2 - \alpha^\top De - \beta^\top D^\top e -\Lambda \circ \mathbf I \bullet D\\
\mathsf{KKT:} & \\
& \nabla F+D - ae^\top - e\beta^\top -\Lambda \circ \mathbf{I} = 0 \\
& \Lambda \ge 0
\end{aligned}$$


Suppose at iteration $k$ projected gradient $D_k = 0$, then the KKT condition for 

We relax one condition for active inequality for some $e = (i,j), e \in M$ such that $X_e =0$, a new optimal direction for problem PD is achieved at $\hat D$, we have:

$$\begin{aligned}
 & \hat D_{ij} - (\alpha_i + \beta_j) + (\hat \alpha_i + \hat \beta_j) - \Lambda_{ij} = 0, \quad e = (i,j) \\
\end{aligned}$$



# Computational Results

The experiments are done on dataset of [QAPLIB](http://anjos.mgi.polymtl.ca/qaplib/), also see paper [@burkard1997qaplib]

The $\mathscr L_2$ penalized [formulation](#mathscr-l_2--mathscr-l_1-penalized-formulation) with code `qap_lp.models.qap_model_l2.l2_exact_penalty_gradient_proj`, solved by gradient projection of module `qap_lp.models.qap_gradient_proj` can solve almost all instances, except for very large ones ($\ge$ 256), it should be better since it now uses Mosek as backend to solve orthogonal projections, line search for step-size, and so on.

current benchmark


```{.table caption="L_2 + L_1 penalized gradient projection" source="l2_grad_proj_benchmark.20201012.csv"}  
```

# Reference