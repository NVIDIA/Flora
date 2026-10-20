---
library_name: peft
base_model: Qwen/Qwen3.5-9B
tags:
- peft
- lora
- qwen3.5
- video
- temporal-segmentation
- data-engine
---

# Flora-segmenter-9B

# Model Overview

### Description:

Flora-segmenter-9B localizes operations in robot and human manipulation videos. For each input window, it predicts local interval coordinates and flags indicating whether an operation continues beyond the visible window. The Flora data engine projects these fragments onto the source clock and stitches them into episode-level events; the separate Flora captioner generates their action instructions. Flora was developed by NVIDIA as the data engine used for training GR00T-2, and is released to help researchers scale up custom robot datasets.

This model is for research and development only.

### License/Terms of Use:
Use of this model is governed by the [NVIDIA OneWay Noncommercial License (22 March 2022)](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf).<br>
This model has been post-trained on `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The Qwen base model is licensed under [Apache License 2.0](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/LICENSE).<br>
You are responsible for ensuring that your use of NVIDIA provided models complies with all applicable laws. <br>

### Deployment Geography:
Global

### Use Case:
Researchers, Academics, Open-Source Community: Annotating robot datasets with sub-task segments; and research into emerging applications such as physical auto-research and human-to-robot transfer.<br>
Developers: Prototyping and evaluating data curation, filtering, and monitoring pipelines for robotics foundation model research.<br>

### Release Date:
- **Hugging Face:** 10/15/2026 via [Flora-TempSeg](https://huggingface.co/nvidia/Flora-TempSeg).

### Intended Use

Flora is intended for non-commercial robotics research and development only, including annotating robot and human demonstration videos. Flora is not intended or supported for commercial or production use. Flora outputs are annotations. It is not intended to directly control robots or to serve as the sole safety monitor for a physical system.

## References(s):
* GR00T Website: [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/gr00t)
* GR00T N1 White Paper: https://arxiv.org/abs/2503.14734
* Base Model: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)<br>

## Model Architecture:
**Architecture Type:** Transformer

**Network Architecture:** Other: Qwen3.5 vision-language model (natively multimodal vision encoder with a decoder-only Large Language Model)

**This model was developed based on Qwen3.5-9B.**

**Number of model parameters:**
* Flora-segmenter-9B: 9.0 x 10^9

Flora-segmenter-9B was trained with Low-Rank Adaptation (LoRA) on the language model's attention and MLP projections, with rank 32 and alpha 64; the base weights, including the visual encoder, remain frozen. Training has two stages: supervised fine-tuning, followed by reinforcement learning using Group Relative Policy Optimization (GRPO) with a verifiable temporal-matching reward that updates only the same LoRA parameters. Sampled video frames are encoded by the vision encoder. The resulting visual tokens are combined with a prompt specifying the window geometry, emission core and annotation policy, and passed to the language model decoder. The decoder autoregressively generates a `segments` JSON object containing local interval coordinates and continuation flags, without action labels.



## Input(s):
**Input Type(s):**
* Video: Ordered Red, Green, Blue (RGB) frames for one local temporal window, from one camera or three synchronized cameras<br>
* Text: Prompt specifying the window's valid range, emission core, local time coordinates and annotation policy<br>

**Input Format(s):**
* Video: Decoded RGB frames from a video file (e.g., .mp4), selected using its source timestamps<br>
* Text: String<br>

**Input Parameters:**
* Video: Four-Dimensional (4D): a sequence of RGB images (frames × height × width × 3), with one image or synchronized camera mosaic per temporal slot<br>
* Text: One-Dimensional (1D)<br>

**Other Properties Related to Input:** Native reference inference uses the trained window profiles below. Select the corresponding trained prompt and window envelope together.

| Profile | Window duration | Sampling | Frame slots | Emission core | Context guards |
| --- | --- | --- | --- | --- | --- |
| Standard | 16 seconds | 4 Hz | 64 | 12 seconds | 2 seconds on each side |
| Long | 32 seconds | 4 Hz | 128 | 24 seconds | 4 seconds on each side |

For a standard interior window, the visible local range is `[0, 320)` in 20 Hz ticks, and the emission core is `[40, 280)`. For a long interior window, they are `[0, 640)` and `[80, 560)`. Consecutive standard windows can advance by the 12-second core width, sharing context; the production pipeline can also request shifted windows to resolve uncertain boundaries. At the beginning or end of an episode, record the actual valid and core ranges in the prompt instead of treating padded frames as new observations.

Select the nearest available source frame for each scheduled slot using the source timestamps. Edge padding repeats the nearest real sampled frame; slots outside the valid envelope are not evidence. The processor receives the prepared slots with the profile's local sampling clock and must not resample them (`do_sample_frames=False`). This 4 Hz sampling clock is distinct from both the video's original frame rate and the 20 Hz output coordinate system.

For single-camera input, use ordinary RGB frames. For three-camera input, tile synchronized views at each source time into one mosaic: main camera above, left-wrist camera below left, right-wrist camera below right. Keep panel roles and layout fixed and state them in the prompt. Native temporal preprocessing preserves aspect ratio and caps each single-camera frame at 40,000 pixels, or each three-camera mosaic at 160,000 pixels, before processor resizing. Use the trained Qwen3.5 processor and chat template with a 32,768-token sequence budget; the native processor uses `min_pixels=1024` and `max_pixels=589824`.

### Time coordinates and example prompt

One output tick represents 0.05 seconds relative to the requested window origin `t0`. Project a local coordinate `k` back to the source clock with `source_time_s = t0 + k / 20`. Keep the original source timestamps and window origin for this projection. Tick granularity specifies the output representation and does not guarantee 50 ms boundary accuracy.

The following illustrates the standard interior-window prompt. The `<video>` marker stands for the trained processor's video token with 64 ordered RGB frames attached. For three-camera input, also prepend the camera-mosaic description above.

```text
<video>
The video contains 64 ordered temporal slots sampled at 4 Hz for one 16-second local view.
The emit core is 12 seconds with a 2-second guard on each side.
Segment meaningful local operations. Acquisition and delivery are distinct when observable;
retain distinct directional handovers. Ordinary contacts, adjustments and low-motion pauses
within the same ongoing goal do not create new events.
Edge padding repeats the nearest real sampled frame; ignore slots outside the valid tick envelope.
Times use 20 Hz local ticks in the half-open range [0,320).
Canonical window envelope:
{"valid_start_tick":0,"valid_end_tick":320,"core_start_tick":40,"core_end_tick":280}
Emit only intervals with positive half-open intersection with [core_start_tick,core_end_tick).
Clip intervals at valid video limits; lo/ro indicate continuation beyond the visible limit.
lo=true requires s==valid_start_tick; ro=true requires e==valid_end_tick.
Return exactly one JSON object and no prose or Markdown, with only top-level key segments.
Each row has keys exactly in order s,e,lo,ro. Sort rows by (s,e,lo,ro).
```

Use the annotation-policy text accompanying the selected trained profile; the example illustrates the geometry and serialization contract.

## Output(s)
**Output Type(s):** Text<br>
**Output Format:** String containing one JSON object<br>
**Output Parameters:** One-Dimensional (1D)<br>
**Other Properties Related to Output:**
* Flora-segmenter-9B: `{"segments": [...]}`, with integer local ticks `s` and `e`, and Boolean continuation flags `lo` and `ro` for each half-open interval `[s, e)`.
* `lo=true` means the operation continues beyond the left visible limit and requires `s == valid_start_tick`; `ro=true` means it continues beyond the right visible limit and requires `e == valid_end_tick`.
* All intervals must lie within the valid envelope and have positive overlap with the emission core. Distinct concurrent events may overlap; do not force them into a non-overlapping partition.

**Illustrative output for the standard window above:**

```json
{"segments":[{"s":40,"e":120,"lo":false,"ro":false},{"s":200,"e":320,"lo":false,"ro":true}]}
```

If `t0 = 60` seconds, these fragments project to `[62, 66)` and `[70, 76)` in source seconds. The second operation continues beyond the visible right limit, so 76 seconds is the observed fragment endpoint rather than a verified operation end. Reconstruct complete events by matching compatible fragments across overlapping windows, preserving continuation flags and concurrent events. Apply the separate captioner to each reconstructed target interval to obtain its action instruction.

Our AI models are designed and/or optimized to run on NVIDIA Graphics Processing Unit (GPU)-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Runtime Engine(s):** PyTorch (Hugging Face Transformers)

**Supported Hardware Microarchitecture Compatibility:**
* NVIDIA Ampere
* NVIDIA Blackwell
* NVIDIA Hopper
* NVIDIA Lovelace

**Supported Operating System:**
* Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology (a development process that pairs each design stage with a corresponding testing stage), iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment. <br>

## Model Version(s):
* Flora-segmenter-9B (v1): Sub-task temporal segmentation; fine-tuned from Qwen3.5-9B with supervised fine-tuning followed by reinforcement learning. It integrates into an AI system as the segmentation stage of the Flora data-annotation pipeline: the pipeline splits an episode into overlapping video windows, runs the model on each window, projects the predicted local intervals onto the source clock, and stitches them into episode-level sub-task events that can then be captioned by Flora-captioner-27B.

## Training, Testing, and Evaluation Datasets:

### Training Data Summary

Training labels are a combination of human-written and VLM-generated annotations.

## Training Data Sources:
Training data is a combination of open-source datasets, datasets licensed from vendors, and data collected in-house by NVIDIA.<br>

## Training Dataset:

** Data Modality <br>
* [Video] <br>
* [Text] <br>

** Video Training Data Size <br>
* Less than 10,000 Hours<br>

** Text Training Data Size <br>
* Less than a Billion Tokens<br>

** Data Collection Method by dataset <br>
* [Hybrid: Automatic/Sensors, Manually-Collected]<br>

** Labeling Method by dataset <br>
* [Hybrid: Manually-Labeled, Automatic/Sensors, Synthetic]<br>

**Properties:** <br>
Robot camera video from multiple embodiments and human egocentric video of hand-object manipulation, paired with sub-task segments. Less than 10 million training examples, covering less than 10,000 hours of video. All training text is in English.

### Testing Dataset:

** Data Collection Method by dataset:
* Not Applicable

** Labeling Method by dataset:
* Not Applicable

** Properties:
* Not Applicable. No separate testing dataset was used, so data count, modalities, content, and linguistic characteristics do not apply. The model was evaluated on the benchmarks listed under Evaluation Dataset, using the test hardware listed under Inference.

### Evaluation Dataset:

** Data Collection Method by dataset:
* Hybrid: Automatic/Sensors, Manually-Collected

** Labeling Method by dataset:
* Hybrid: Manually-Labeled, Automatic/Sensors

**Properties:** The paper evaluates WGO-Bench on 100 human and robot manipulation episodes (25 HomER, 50 DROID, 25 Galaxea), containing 743 gold segments. Segmentation uses the production window-processing and continuation-stitching pipeline, with one-to-one interval matching at temporal Intersection over Union (IoU) >= 0.75. The reported score is the equal-weight average across the three subsets.<br>
* Modalities: Video and text. Linguistic characteristics: English task instructions and English sub-task segment annotations.

**Benchmark Scores (reported in the Flora paper, Table 2):**

| Model | Benchmark | Metric | Flora |
| --- | --- | --- | --- |
| Flora-segmenter-9B | WGO-Bench | Segmentation F1 at temporal IoU >= 0.75, subset average (%) | 50.3 |

## Inference:
**Acceleration Engine:** PyTorch on CUDA (Hugging Face Transformers), using Brain Float 16 (BF16) with 32-bit floating point (FP32) Gated DeltaNet parameters
**Reference Decoding:** Load the LoRA adapter with its Qwen3.5-9B base and trained processor. Use the matching profile prompt, thinking disabled, greedy decoding, and at most 1,024 new tokens per window. Parse and validate the local JSON before source-time projection and stitching.
**Reference Precision:** Native inference preserves FP32 Gated DeltaNet parameters and the saved FP32 LoRA parameters, with BF16 base weights and BF16 temporal compute autocast. Preserve these mixed parameter dtypes when loading the weights and adapter.
**Test Hardware:**
* NVIDIA Hopper H100

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included. <br>

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
