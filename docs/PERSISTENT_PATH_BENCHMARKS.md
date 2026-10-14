# Persistent terrain and connected-path test

Tested on 2026-10-14, before application integration. The controlled connected
path tests pass. The recorded-video tests still fail the acceptance gate for
reliable real-time navigation. Production code and the running server were
unchanged; experimental scripts are outside the application repository.

## What was tested

The prototype retains a rolling 200 x 200 grid of actual observed ground points,
with source frame/row identity, separate base costs and negative veto state.
Evidence and source-specific slope caches are limited to 20 packets. It applies
clearance inflation once after fusion and estimates candidate slopes separately,
without accumulating slope penalties on retained costs. It uses the existing
native bounded grid search and original raw support-gap and swept-obstacle
predicates. Proposed slope-enabled route nodes also receive the original robust
fit against geometry from their own source frame.

Two route policies were tested:

- **Current-ground suffix replan:** require an eligible point within 1.5 cells
  horizontally of the camera projection, then associate it with the retained
  proposed route within two cells. Preserve the prefix through that anchor and
  replace the future suffix. Off-route camera motion returns unavailable.
- **Tail extension demo:** begin on visible ground and extend from the previous
  endpoint. This is a continuity experiment; its goals are relative to that
  endpoint, not the current camera. It does not satisfy the requested
  camera-relative moving-goal policy by itself.

Every accepted splice shares an identical world coordinate with the retained
path. The newly proposed suffix, including any connector to a refreshed cell,
must pass raw geometry validation. A rejected update publishes no active route;
the previously proposed polyline remains historical. That history is not
asserted currently traversable or physically executed.

Current-ground association here is only a planar camera projection. It does
not verify a calibrated robot-ground pose or the swept connection from a robot
body to the ground sample. Recorded camera movement is not controller feedback.

## Controlled replay

A changing near-field view observes a known plane. The simulated camera advances
along the proposed route, a negative obstacle appears on the future route, and
the heading later turns by 90 degrees. Both slope settings produced one initial
route followed by **11 connected suffix replans across 12 updates**. Both
detoured around the new obstacle and preserved the prefix through every anchor.

| Controlled replay | Median CPU | P95 CPU | Maximum CPU |
|---|---:|---:|---:|
| Slopes disabled | 32.4 ms | 65.0 ms | 101.4 ms |
| Slopes enabled | 65.9 ms | 74.2 ms | 75.3 ms |

These short-run measurements include evidence indexing, grid construction,
fusion, search and validation; projected input generation is excluded. They
are not sustained throughput or end-to-end perception timings.

Eight independent geometry/state checks also passed:

1. A supported unobstructed corridor is accepted.
2. An unknown ground gap rejects a proposed join.
3. A negative wall rejects a join despite overlapping positive support.
4. An overhead obstacle rejects the swept body-clearance test.
5. A 45-degree ramp is rejected with slopes enabled and accepted with them off.
6. Slopes off still preserve the negative-obstacle veto.
7. A camera off the proposed route receives no artificial connection.
8. A splice cannot rewrite an already committed prefix.

Additional replay assertions reject mismatched joins, keep negative vetoes when
later positives arrive, and return unavailable after ground witnesses expire.
The obstacle/turn tests use explicit simulated progress, not real robot control.

## Recorded-video replay

Chronological replay covered the existing 56 mapped IMG_5701 frames and six
IMG_5516 frames under each of the VLM-negative and VLM-positive configurations,
including independent SAM `path` positives. The two IMG_5516 configurations
reuse geometry. This is 62 unique mapped geometry frames from two clips,
68 configuration-specific updates, and 272 route-policy/slope evaluations.

| IMG_5701 policy | Slopes | Initial routes | Connected updates | Unavailable / 56 | Median CPU on accepted updates | Maximum accepted CPU |
|---|---|---:|---:|---:|---:|---:|
| Current-ground suffix replan | Off | 1 | 12 replans | 43 | 128.2 ms | 135.8 ms |
| Current-ground suffix replan | On | 1 | 1 replan | 54 | 283.3 ms | 357.0 ms |
| Tail extension demo | Off | 1 | 32 extensions | 23 | 54.8 ms | 137.7 ms |
| Tail extension demo | On | 1 | 3 extensions | 52 | 171.6 ms | 340.4 ms |

No artificial join or prefix rewrite was accepted. This does not make the many
unavailable updates successful navigation outcomes.

In the current-ground cases, missing ground association accounted for 14/15
updates with slopes off/on; the camera moving off the proposed path accounted
for 27/37. Off-route movement is expected to be possible in an independently
recorded video, so this experiment cannot establish robot execution success.
However, it also shows that the application cannot simply infer progress from
camera timestamps or draw links between unrelated frame paths.

The slope-enabled tail demo lost its anchor to blocking/expiry on 18 updates
and outside the rolling window on 19. Fourteen updates had no forward goal;
one suffix failed raw geometry validation. It therefore does not sustain the
requested continuous route.

The IMG_5516 cases supplied no strict current-ground start in any of their six
updates. Each demo obtained one initial path but no accepted extension. Short
lookahead/observability and retained-anchor validity still need an explicit
production policy.

Source-aware robust slope checks made **397 comparisons** against previously
computed original same-frame values. Every finite/NaN pattern matched and the
maximum numerical difference was zero. These include repeated queries, not
397 distinct terrain points. This checks source correspondence at queried
representatives, not exact equivalence of the entire fused support graph.

## Performance interpretation and remaining work

Low medians over all updates hide the cost of actually producing a valid path.
For example, slope-enabled current-ground processing had a 55.5 ms median
across all 56 updates, but its only two accepted routes took 209.6 and 357.0 ms.
The 100 ms combined CPU target is still missed on successful temporal updates.

For those two accepted updates, raw geometry validation took 120.0/140.6 ms,
and source-specific slope refinement/index creation took 7.6/134.7 ms.
Reconstructing validation structures from corridor witnesses and creating raw
KD indexes lazily remain material costs. Native graph search is not the main
remaining optimization target. The fixed limits were 40,000 grid cells and
20 evidence packets; this is not a long-duration memory or latency test.

The next implementation gates remain:

- Verify current-ground pose and actual progress before committing history.
- Retain stable anchors with sufficient source-specific witnesses; distinguish
  stale terrain, negative hazards, reference changes and real geometry conflicts.
- Choose camera-relative observed goals and replace future suffixes when
  obstacles/heading change, instead of blindly extending the last target.
- Bound witness validation and source-index work without dropping relevant
  negatives or relaxing support/clearance checks.
- Measure a paced live-model replay and long-run behaviour before claiming real
  time. Supplied calibration and resident projected data exclude model startup,
  inference, projection, online reference estimation and rendering.

The experimental negative-veto expiry of 20 packets is not a production hazard
clearing policy. Reference changes require explicit replacement of terrain and
path state. Neither this test nor the camera-relative model establishes metric
robot footprint/kinematic feasibility.

## Artifacts and reproduction

Artifacts are in `/home/jovyan/EmbodiedAI-gpu-validation/framewise-benchmarks/`:
`persistent_connected.py`, `persistent_guards.py`, `persistent-report.json`,
`persistent-samples.json`, `persistent-guards-report.json` and `persistent.log`.
The manifest records source/report hashes; CSV timing rows are also retained.
The previous [framewise benchmark report](FRAMEWISE_BENCHMARKS.md) documents
input snapshots, calibration and the shared search/geometry helpers.

```bash
cd /home/jovyan/EmbodiedAI-gpu-validation/framewise-benchmarks
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/jovyan/EmbodiedAI-integration/.venv-integration/bin/python \
  persistent_connected.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/jovyan/EmbodiedAI-integration/.venv-integration/bin/python \
  persistent_guards.py
```

Application server PID 13137 remained running on port 8083 throughout testing.
No GPU model worker was started by these CPU experiments.
