# CPU analysis performance

Validated using the saved live IMG_5701 map with 6,247,214 original rows and
3,451,693 finite-cost rows, in a container with a 24-core CPU allowance.
The improved analysis uses 12 CPU threads with one BLAS thread per operation.

| Stage | Original observed interval | Improved measured runtime |
|---|---:|---:|
| Steepness | 275 seconds | 31 seconds |
| Costs, including baseline clearance calibration | 64 seconds | 55 seconds |
| Exact route and final edge verification | 499 seconds | 344 seconds |

Original intervals are measured between the map, steepness and cost receipts;
they include stage IO and reference fitting. Improved runtimes are direct wall
timings while the original live run was still searching for its route. Together,
the preprocessing stages decreased from about 5.7 minutes to 1.4 minutes on this
map. Including exact route search, total analysis decreased from about 14 minutes
to 7.2 minutes. This is an overall improvement of about 2×, with about 9× faster
steepness; exact optimality proof remains the dominant cost.

Every slope, quality and cost row was compared with the original saved arrays:
all values, NaN locations and blocked rows matched exactly. Camera-clearance
height and its 75 calibration observations also matched. Independent regression
tests cover shuffled frames, duplicates, invalid points, negative vetoes, cost
boundaries, connection IDs and fallback to the complete graph when a sampled
native search fails.

The improved exact route also matched the original **211 row IDs, coordinates
and total cost exactly**, with cost 1.7782178281790924. The artifact viewer loaded
Source, Traversability, Steepness and Cost map modes and the complete verified
route without provenance warnings.

Parallel work retains the same distinct-neighbour selection, robust plane fit,
support tests and swept clearance checks. Native feasible-bound search avoids
the Python loop that repeatedly reconstructs sampled graphs and rejects invalid
candidate edges. Its bound is independently validated before pruning the full
graph; exact route certification and partial-route rules are retained.

The automatic worker count respects CPU affinity and the cgroup v2 quota and
is capped at 12 to bound memory usage. The numerical slope/cost APIs keep their
serial default, while artifact pipelines and the live viewer select automatic
parallel execution. Single-frame numerical slope calls remain one fitting job.

Regression validation: **225 tests pass** (161 planner, 44 plugin and 20 integration).

The original exact route stage took 499 seconds and returned an optimal
211-point route with cost 1.7782178281790924. The improved native preliminary
search found an independently validated feasible bound in about four seconds;
full optimality proof is measured separately.

Machine-readable measurements and full-array comparisons are saved under
`/home/jovyan/EmbodiedAI-gpu-validation/performance-final`. Route search runtime
is recorded separately; it can still be substantial on dense or disconnected
graphs. Steepness and cost speedups do not imply the same factor for route search.
