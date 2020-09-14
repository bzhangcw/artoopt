---
# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /model.md
# @created: Wednesday, 9th September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Wednesday, 9th September 2020 2:42:09 pm
# @description: 


---

## the problem

QAP,

$$\begin{aligned}
&\min_X \textrm{tr}(A^\top XB^\top X)  \\
\mathbf{s.t.} & \\ 
&X \in \Pi_{n}
\end{aligned}$$

is the optimization problem on permutation matrices:

$$ \Pi_{n}=\left\{X \in \mathbb R ^{n \times n} \mid X e =X^{\top} e = e , X_{i j} \in\{0,1\}\right\}$$

The convex hull of permutation matrices, the Birkhoﬀ polytope, is defined:

$$D _{n}=\left\{X \in \mathbb R ^{n \times n} \mid X e =X^{\top} e = e , X \geq 0 \right\}$$

