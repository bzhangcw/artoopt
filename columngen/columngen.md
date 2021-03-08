---
bibliography: [../ref.bib]
title: Column generation
---

The following formulation, notation repeats from, @desrosiers_primer_2005.


## Capacitated Shortest path

For graph $\mathcal G(V, A)$
$$\begin{aligned}
z^{\star}:= &\min \sum_{(i, j) \in A} c_{i j} x_{i j} \\
&\text { s.t.} \\
&\sum_{j:(s, j) \in A} x_{s j}=1 \\
&\sum_{j:(i, j) \in A} x_{i j}-\sum_{j:(j, i) \in A} x_{j i}=0 &\quad i \in I \backslash\{s, t\} \\
&\sum_{i:(i, t) \in A} x_{i t}=1 \\
&\sum_{(i, j) \in A} t_{i j} x_{i j} \leq C \\
&x_{i j}=0 \text { or } 1 \quad(i, j) \in A
\end{aligned}$$


with $p \in P$ as set of feasible paths:

$$\begin{aligned}
z^{\star}=&\min \sum_{p \in P}\left(\sum_{(i, j) \in A} c_{i j} x_{p i j}\right) \lambda_{p} \\
\text { s.t. } &\\
&\sum_{p \in P}\left(\sum_{(i, j) \in A} t_{i j} x_{p i j}\right) \lambda_{p} \leq C \\
&\sum_{p \in P} \lambda_{p}=1 \\
&\lambda_{p} \geq 0 \quad p \in P \\
&\sum_{p \in P} x_{p i j} \lambda_{p}=x_{i j} \quad(i, j) \in A \\
&x_{i j}=0 \text { or } 1 \quad(i, j) \in A
\end{aligned}$$


Remark:

- The problem without the capacity constraint is a regular shortest path problem. Any path is an extreme point for convex hull defined by integral solutions.
- While the cardinality $|P|$ may be prohibitive, the C-G procedure can be used for this.


dual with $\pi_0, \pi_1$:



