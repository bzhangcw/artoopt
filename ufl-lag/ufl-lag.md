---
bibliography: [../ref.bib]
title: Uncapacitated Facility Location, Lagrangian Relaxation
---
# Formulation 

$$\begin{aligned}
\max_{x, y} & \sum_{i=1}^{m} \sum_{j=1}^{n} c_{i j} y_{i j}-\sum_{j=1}^{n} f_{j} x_{j} \\
&\sum_{j=1}^{n} y_{i j}=1, & \forall i \\
& y_{i j} \leq x_{j}, & \forall i, j\\
& x \in\mathscr B^{n}, y \geq 0 \\ 
\end{aligned}$$

Compact:

$$\begin{aligned}
\max_{x, y}\; & C\bullet Y - f^\top x \\
&Y e= e, & \forall i \\
& y_{i j} \leq x_{j}, & \forall i, j\\
& x \in\mathscr B^{n}, y \geq 0 \\ 
\end{aligned}$$

Lagrangian relaxation on first set of constraints: $\lambda \in \mathbb R^m$

$$\begin{aligned} 
z^{L R}(\lambda)=\max_{x, y}\; & C\bullet Y - f^\top x  + e^\top \lambda - \lambda^\top Ye\\ 
& y_{i j} \leq x_{j} \\ 
& x \in\mathscr B^{n}, y \geq 0 
\end{aligned}$$

Analytic solution:

$$\begin{aligned} 
y_{i j}(\lambda)=&\left\{\begin{array}{ll}1 & \text { if } c_{i j}-\lambda_{i}>0 \text { and } \sum_{\ell}\left(c_{\ell j}-\lambda_{\ell}\right)^{+}-f_{j}>0, \\ 0 & \text { otherwise. }\end{array}\right.\\ & x_{j}(\lambda)=\left\{\begin{array}{ll}1 & \text { if } \sum_{\ell}\left(c_{\ell j}-\lambda_{\ell}\right)^{+}-f_{j}>0 \\ 0 & \text { otherwise. }\end{array}\right.\end{aligned}$$

$\nabla Z^{LR}_\lambda = e - Y^\star e$, where $Y^\star$ is the optimal solution of the $z^{LR}(\lambda)$
