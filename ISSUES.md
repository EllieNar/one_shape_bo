# ISSUES

Problems with the legacy code. Not in any particular order.
Each issue should be explored in isolation unless specifically dirceted otherwise.

## ISSUE 01
### Whether the (near) global optimum is found heavily depends on the burn-in.

A burn-in set dominated by extreme geometries may result in strong local optimal which reduce exploration in later iterations.
In the current legacy code, (see Sampling > random_shape_candidate), a prior is applied on the shape area selection.
This distributes the area roughly equivalently during burn-in, but allows area exploration during optimisation.
However, the latter does not appear to be happening, and the solution exhibits even worse convergence
to the global optimum. Perhaps lack of diversity in the initial dataset was too extreme.

Possible solutions:
- The weighting associated with Burn_In might be too strict.
Perhaps try permitting a slightly greater area variation, or even
weights very similar to BO_Eval, but so that very extreme hole areas are avoided.

## ISSUE 02
### Exploration drops after ~ 20 * design_dimension iterations.
Perhaps the local optima become too strong, is the local radius search too small?

Possible solutions:
- Increase ng_rst and reduce nl_rst; perhaps the local optimum becomes too dominant?

## ISSUE 03
### Improper coverage of the design space at burn-in
The current sampling method is feasibility-driven, not space-filling.
Rejection probebilities depend on shape, size and placement history.
The sequential place-and-freeze provedure can bias the distribution towards
configurations in which the early shapes are easy to place. It therefore
does not reliably cover the continuous design space as well as e.g.
Latin Hypercube Sampling or Sobol sampling methods might permit.

Possible solutions:
- See the legacy code Functions > Burnin. This underdeveloped code provides
a foundation for a space-filling DOE. However, it stands to be very computationally
inefficient: very large LHS pools are generated and optimised, while
many rows are subsequently lost to containment and later, geometric constraints
(such as overlap, amin, smin etc). The overlap constraint is also coupled and
non-sequential, so simply filtering or hoping that LHS rows are feasible does
not guarantee a large feasible, well-spaced burn-in set.


