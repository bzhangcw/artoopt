---
bibliography: [./optlayer.bib]
title: Optimization as a Layer
date: \today
mainfont: Roboto
---


# Optimization as a layer

## Some Concepts

(.) is the alias:

- (**e2e**) End-to-end: raw-input to ultimate outputs of interests.
  - without tuned features/feature engineering
- (**optlayer**) Optimization as a Layer
  - inputs $\to$ outputs: $x\to y$, includes an optimization problem $y = \arg \min f(x)$
- Differentiation,  forward-and-backward pass.
  - for differentiable (convex) problem, Jacobian can be calculated via KKT.
  - for LP, can be calculated via interior point HSD formulation, @ye_o_1994
  - other specified solvers, for QP, conic, ...


## Differentiations

[optimal condition] + solver

- KKT + QP solver, @amos_optnet_2017
- CVX -> Conic (HSD embedding) [optimal condition] -> conic solver (SCS, ...), @amos_optnet_2017
- LP -> HSD, @mandi_interior_2020

## Application

- e2e, stochastic programming (single stage.)
  - sp -> deterministic.

- sensitivity analysis
- 





## Reference

<!-- Q
  - not solve to optimal?
  - 
 -->
