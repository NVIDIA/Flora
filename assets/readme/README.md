# README media

`flora-film-teaser.webp` is a 12-second excerpt of the Flora launch film, the same film as the
[project page](https://research.nvidia.com/labs/gear/flora/): the title, a timestamped instruction for a
guitar-repair video, a dense-reward failure ("drops the white packet") and an outcome score of 1. The
on-screen outputs are recorded Flora predictions, and each agrees with a human label: the instruction
matches the human annotation of that interval ("pick up a white bridge pin from the gray container … with
the right hand"), the dense failure (predicted 19.4–21.4 s) overlaps the human-labeled failure
(19.3–20.8 s), and the outcome score equals the human label (1).

The caption and outcome/rubric samples are resized copies of short video loops already published
on the [FLORA project page](https://research.nvidia.com/labs/gear/flora/).
The accompanying captions and rewards in the repository README are the
project page's published predictions. Outcome and rubric scores refer to the
full episodes; the animations show short excerpts.

| Asset | Published source |
|---|---|
| `caption-human.gif` | [Human example](https://research.nvidia.com/labs/gear/flora/assets/hero/bubbles/clip_10.mp4?v=6) |
| `caption-parallel.gif` | [Parallel-gripper example](https://research.nvidia.com/labs/gear/flora/assets/hero/bubbles/clip_33.mp4?v=6) |
| `caption-dexterous.gif` | [Dexterous-hand example](https://research.nvidia.com/labs/gear/flora/assets/hero/bubbles/clip_02.mp4?v=6) |
| `reward-outcome.gif` | [Outcome example](https://research.nvidia.com/labs/gear/flora/assets/hero/bubbles/clip_31.mp4?v=6) |
| `reward-rubric.gif` | [Rubric example](https://research.nvidia.com/labs/gear/flora/assets/hero/bubbles/clip_90.mp4?v=6) |

`reward-process.gif` shows a held-out DenseRewardBench episode (toy-car assembly, front camera, cropped,
episode time 189.4–193.4 s). Flora scores 188.2–199.9 s as a failure: "the right hand fails to tighten the
screw using the toy drill as the alignment between the toy drill and the screw is imprecise. the toy drill is
misaligned and contacts the gray screw head without engaging the screw". All 24 half-second scores match the
human label ("failure", note "due to misalignment"), and the reason was checked against the front and
right-wrist views.

The method figures are unmodified figure-only extracts from the public
[FLORA paper](https://research.nvidia.com/labs/gear/flora/assets/paper/flora.pdf),
rendered as PNGs for GitHub. Surrounding page text and captions are omitted.

| Asset | Paper figure | PDF page |
|---|---|---|
| `figure-7.png` | Video-to-supervision overview | 8 |
| `figure-13.png` | Windowed localization and contextual captioning | 31 |
| `figure-14.png` | Synchronized views and temporal context | 31 |
| `figure-15.png` | Outcome, process, and rubric reward interfaces | 36 |

Timelines and values in the method figures are illustrative. They describe
inference interfaces, not training settings or production infrastructure.
