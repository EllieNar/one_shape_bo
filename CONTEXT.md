## CONTEXT.md

# This file is intended to record:
- A concise summary of what solutions/approaches were explored in latest query
- Durable project decisions
- Accepted engineering assumptions
- Validation results
- Unresolved issues

# Every entry should begin with the following formatting:
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 		            DD / MM / YY  - TIME
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                                            22/09/26    at    14:52
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

New One_Shape_Only folder with 260921_Topology_Optimisation_ONESHAPE.py

In comparison to 260820_Topology_Optimisation_MAIN, this has the following changes:

1. No 'improvement count': there is no 'if improvement_count >= 75' so that the code can continue way beyond when it might have otherwise concluded. This ensures whether it can escape local minima in later iterations. As per meeting on 21/09/26 (see blue book), S suggested that the plateau can see in the log-loss graph (see 260911_Meeting5.ppt, slide 6) is because there is not enough exploration in the middle iterations (and thus it is getting stuck in local minima).

2. Changed fixed_features in sampling so that it handles 'htype' which dictates whether **all** holes are triangular or whether **all** holes are elliptical (rather than a random combination of triangle/ellipse). This is because of previous issues with improper sampling of the design space (trying to move to LHS rather than a feasibility-driven sampler). The change from a feasibility-driven sampler to something LHS-based is an investigation required to ensure fair examination of the design space when both triangular and elliptic holes are present.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                                            23/09/26     at      16:32
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

In random_shape_candidate, weights are changed so that all areas have similar areas rather than randomly picking weights that can vary between 0 and 1.

Must increase batch_size, max_batches, anchor_trials and shape_trials in sample_random_feasible to generate enough burn-in (concerned with triangles).

Even still, the code is very slow.

FYI this is following the proposed change in the meeting with B and S on 21/09/26. S suggested two changes to get closer to the global optimum:

1. Let the code continue for longer (see previous date's log)
2. Let the algorithm explore larger perforations (i.e. after burn-in). Currently, the algorithm may be getting stuck in strong local minima because of residue large perforations that are echos from the burn-in.

Starting with triangular holes because need to try and get the same/similar result as see in the folders below (before can move onto assuming that the ellipse-only code gives a correct result):

Folder 1 (using the triangle only code): C:\Users\eleno\OneDrive - Nexus365\Documents\POSTDOC\Code\Topology_Optimisation\Bayesian_Optimisation\Outputs\260813_Test
Folder 2 (using the one_shape_only code, which could accommodate either all triangles or all ellipses) : C:\Users\eleno\OneDrive - Nexus365\Documents\POSTDOC\Code\Topology_Optimisation\Bayesian_Optimisation_Part2\Outputs\One_Shape\Triangle\260923

The results are again stuck in a local optima. Either:
1. Not exploring domain enough - try with n global restarts outnumbering n local?
2. Ask Codex to improve exploration around midway through the iterations?