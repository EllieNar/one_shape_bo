# ISSUES

Problems with the legacy code. Not in any particular order.
Each issue should be explored in isolation unless specifically dirceted otherwise.
The expected output for three triangular holes, when the code 260928_MBB_Beam_Basic.py is
run is within the outputs/Expected_Output_TRIANGLE folder of this repo. The expected output should be treated as a qualitative topology target until it is converted to the same binary, volume-constrained, three-triangle problem

## ISSUE 01
### Whether the (near) global optimum is found heavily depends on the burn-in.

A burn-in set dominated by extreme geometries may result in strong local optimal which reduce exploration in later iterations.
In the current legacy code, (see Sampling > random_shape_candidate), a prior is applied on the shape area selection.
This distributes the area roughly equivalently during burn-in, but allows area exploration during optimisation.
However, the latter does not appear to be happening, and the solution exhibits even worse convergence
to the global optimum. Perhaps lack of diversity in the initial dataset was too extreme.
In CONTEXT.md read the entry on 25/09/26 and 27/09/26, compare to the expected output.

Possible solutions:
- Re-evaluate the balance between global exploration (ng_rst), local exploitation (nl_rst),
and the area prior. Is the area prior necessary/correct?

## ISSUE 02
### Exploration drops after ~ 20 * design_dimension iterations.
Perhaps the local optima become too strong, is the local radius search too small?

Possible solutions:
- Increase ng_rst and reduce nl_rst; perhaps the local optimum becomes too dominant? (Explored, see CONTEXT.md)
- Variation in ng_rst and nl_rst after n number of evaluations?

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

## ISSUE 04
### Local exploitation is suppressed

1. In 260928_Implemented_Corrections, raw_local is typically much less than raw_global. Why? Surely with a raw_pool_multiplier of 3, it should be around 15 each time.
2. Are the hyperparameters certainly taken as a warm-start? I would have expected less variation in uncertainty as evaluations progress, if this were the case.
3. In the output log, from iterations 140 to 271 (260928_Implemented_Corrections), the compliance of each resulting geometry is significantly greater than the current best. It seems that exploration occurs at the detriment of exploitation.


In each iteration, the pool of local restarts is too small. Why? Is this related to the number of local restarts, and how can it be increased?

Ensure that Codex is NOT simply using the expected output to guide optimisation.
