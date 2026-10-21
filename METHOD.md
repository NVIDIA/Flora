# Method

## Localize actions, then describe them

The segmenter processes overlapping windows and stitches their fragments into
episode-level event intervals. Each interval is then captioned independently
using visual context from before, during, and after the event. Concurrent
actions can retain overlapping intervals.

<a href="assets/readme/figure-13.png"><img src="assets/readme/figure-13.png" width="100%" alt="Figure 13: Windowed temporal localization, fragment stitching, contextual captioning, and timestamped action instructions."></a>

*[Figure 13](https://research.nvidia.com/labs/gear/flora/assets/paper/flora.pdf#page=31): Windowed localization and contextual captioning.*

For multi-camera captioning, synchronized views are tiled into one observation
at each source timestamp. Observations stay in temporal order within the
before, target, and after blocks; unavailable context is omitted.

<a href="assets/readme/figure-14.png"><img src="assets/readme/figure-14.png" width="100%" alt="Figure 14: Synchronized main and wrist views form mosaics sampled in before, target, and after blocks."></a>

*[Figure 14](https://research.nvidia.com/labs/gear/flora/assets/paper/flora.pdf#page=31): Synchronized views and temporal context.*

## Score outcomes, local actions, or custom steps

One rewarder supports three interfaces: **outcome** (sparse) rewards score task
completion, **process** (dense) rewards locate good and poor execution over
time, and **rubric** rewards assign credit to user-defined steps.

<a href="assets/readme/figure-15.png"><img src="assets/readme/figure-15.png" width="100%" alt="Figure 15: One rewarder produces an episode outcome score, dense quality labels with rationales, or scores for custom rubric steps."></a>

*[Figure 15](https://research.nvidia.com/labs/gear/flora/assets/paper/flora.pdf#page=36): Three interfaces, one reward model. The timeline and scores are illustrative.*

