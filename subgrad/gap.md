---
bibliography: [../ref.bib]
title: Generalized assignment problem, Column generation
---

# Model
$$
\begin{aligned}
\max_x & \sum_{i} \sum_{j} c_{i j} x_{i j} \\
& \sum_{j} x_{i j} \le 1 \\
&\sum_{i} t_{i j} x_{i j} \leq T_j, \quad i=1, \ldots, m \\
& x \in\{0,1\}^{m \times n}
\end{aligned}
$$

or:

$$
\begin{aligned}
\max_X\; & C \bullet X \\
& Xe \le e \\
&e ^\top t \circ X \leq T\\
& x \in \mathscr B^{m \times n}
\end{aligned}
$$

