# Automatic references for the public benchmark

Legacy region-classification workflow. The current image-to-hazard-list task
derives references directly from dataset labels; see [protocol](hazard_protocol.md).

This workflow replaces bulk manual labeling. It evaluates a fixed semantic
avoidance policy using existing RELLIS pixel annotations, conditional on geometry.
Phone videos remain an optional whole-pipeline test.

## Reference policy

Use `rellis_material_v1` consistently in reference generation, VLM prompts and
CLIP mappings. Dirt, asphalt and concrete are semantically permitted, conditional
on geometric feasibility. Obstacles, liquids, mud, rubble and non-support classes
are avoided. This material policy does not establish physical traversability.

| Reference | Original RELLIS IDs |
|---|---|
| traversable | 1 dirt, 10 asphalt, 23 concrete |
| non_traversable | 4 tree, 5 pole, 6 water, 7 sky, 8 vehicle, 9 object, 12 building, 15 log, 17 person, 18 fence, 19 bush, 27 barrier, 31 puddle, 33 mud, 34 rubble |
| unresolved | 0 void, 3 grass; unexpected IDs are validation errors |

Grass does not specify the short/tall distinction needed by the project policy.
Exclude it from automatically referenced classification. Void is unlabelled, not
reference-unknown. Source: [official IDs](https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/main/benchmarks/SalsaNext/train/tasks/semantic/config/labels/rellis.yaml).
Use original image IDs, not this file's LiDAR learning_map.

## Automatic preparation

- Sample 20 development keyframes from 00000 and 20 test keyframes from each of
  00001-00004. Save exact IDs. This gives 100 public frames without phone clips.
- Generate/import SAM masks independently of ground-truth labels. Select up to
  five regions per frame using a deterministic area/location rule, deduplication
  and fixed seed. Record the rule and freeze IDs before inference. Human selection
  or confirmation is optional.
- Validate the aligned ground-truth image dimensions and original IDs. Compute
  non-void coverage, class fractions and annotated avoided-pixel counts per mask.
- For primary references require at least 95% non-void coverage and at least 95%
  dominance of one class among non-void pixels. Assign its mapped label and mark
  complete/valid. These are operational purity criteria, not human mask verification.
- Never assign traversable to a ground-dominated mask containing annotated avoided
  pixels. Flag it mixed and report it separately; majority voting must not erase
  small hazards.
- Unresolved references remain null/pending with an exclusion reason. Report their
  counts without asking the user to label them all. A reference-unknown metric is
  null if no such completed references exist; predicted-unknown rate still applies.

## Compatible artifacts

Keep the existing JSONL/API contracts. Derived annotations may add
`annotation_source: dataset_policy`, `reference_scope: semantic_material_policy`
and `reference_exclusion_reason`. Save `metadata/reference_transfer.jsonl`, keyed
by region_id, with coverage, fractions and avoided-pixel counts. Record policy and
thresholds in run metadata. Reference masks, labels and fractions never enter VLM
inputs. Use only RGB, SAM masks and the common policy for inference.

## Evaluation

Report the existing primary metrics with eligible/excluded counts and class/size
coverage. Add one diagnostic over all selected masks, including mixed masks:
accepted-as-traversable regions containing annotated avoided pixels divided by
all regions containing such pixels. Name it annotated-hazard-overlap acceptance;
it includes SAM failures and is separate from the pure VLM score. Retain pixel
counts so boundary noise and small hazards remain visible.

Export observable class/label disagreements, mixed masks and parsing failures.
Recognition/grounding/policy explanations remain tentative until checked. No bulk
manual categorization; an optional audit of at most 20 examples is sufficient for
a small qualitative discussion. Do not use another VLM as reference ground truth.

Own videos need event timestamps and a few checked key frames for the group test,
not dense frame/voxel labels. Cup/cable accuracy is not established by this outdoor
material-policy benchmark. Missing own videos do not block its stated coverage.
