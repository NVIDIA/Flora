---
library_name: transformers
base_model: Qwen/Qwen3.5-27B
tags:
- transformers
- qwen3.5
- video
- video-captioning
- data-engine
---

# Flora-captioner-27B

# Model Overview

### Description:

Flora-captioner-27B generates one action instruction for a supplied target interval in a robot or human manipulation video. It uses neighboring video context to disambiguate the intended local operation. In the Flora data engine, target intervals come from temporal segmentation and stitching; the captioner supplies the text used for balancing the pretraining data distribution. Flora was developed by NVIDIA as the data engine used for training GR00T-2, and is released to help researchers scale up custom robot datasets.

This model is for research and development only.

### License/Terms of Use:
Use of this model is governed by the [NVIDIA OneWay Noncommercial License (22 March 2022)](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf).<br>
This model has been post-trained on `Qwen/Qwen3.5-27B`, revision `fc05daec18b0a78c049392ed2e771dde82bdf654`. The Qwen base model is licensed under [Apache License 2.0](https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/LICENSE).<br>
You are responsible for ensuring that your use of NVIDIA provided models complies with all applicable laws. <br>

### Deployment Geography:
Global

### Use Case:
Researchers, Academics, Open-Source Community: Annotating robot datasets with language; and research into emerging applications such as physical auto-research and human-to-robot transfer.<br>
Developers: Prototyping and evaluating data curation, filtering, and monitoring pipelines for robotics foundation model research.<br>

### Release Date:
- **Hugging Face:** 10/15/2026 via [Flora-Captioner](https://huggingface.co/nvidia/Flora-Captioner).

### Intended Use

Flora is intended for non-commercial robotics research and development only, including annotating robot and human demonstration videos. Flora is not intended or supported for commercial or production use. Flora outputs are annotations. It is not intended to directly control robots or to serve as the sole safety monitor for a physical system.

## References(s):
* GR00T Website: [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/gr00t)
* GR00T N1 White Paper: https://arxiv.org/abs/2503.14734
* Base Model: [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)<br>
* Benchmarks: [WGO-bench](https://huggingface.co/datasets/macrodata/WGO-Bench)<br>

## Model Architecture:
**Architecture Type:** Transformer

**Network Architecture:** Other: Qwen3.5 vision-language model (natively multimodal vision encoder with a decoder-only Large Language Model)

**This model was developed based on Qwen3.5-27B.**

**Number of model parameters:**
* Flora-captioner-27B: 2.7 x 10^10

Flora-captioner-27B is trained in two stages. Supervised fine-tuning updates all weights of the language model and visual merger, without Low-Rank Adaptation (LoRA), while keeping the visual patch embedding and visual transformer blocks frozen. The supervised model is then further optimized with reinforcement learning using Group Relative Policy Optimization (GRPO) with a reference-grounded caption-quality reward, with the vision encoder frozen. Sampled video frames are encoded by the vision encoder. Visual tokens from explicitly labeled BEFORE, TARGET, and AFTER blocks are combined with the annotation prompt and passed to the language model decoder. The decoder autoregressively generates one instruction for TARGET; context-only operations are not part of the requested output.



## Input(s):
**Input Type(s):**
* Video: Ordered Red, Green, Blue (RGB) frames for a TARGET interval, with optional BEFORE and AFTER context, from one camera or three synchronized cameras<br>
* Text: Prompt identifying the temporal blocks and camera roles and requesting a target-only action instruction; an episode task instruction may also be supplied<br>

**Input Format(s):**
* Video: Decoded RGB frames from a video file (e.g., .mp4), with their source-frame timing metadata<br>
* Text: String<br>

**Input Parameters:**
* Video: Four-Dimensional (4D): a sequence of RGB images (frames × height × width × 3), organized into labeled temporal blocks<br>
* Text: One-Dimensional (1D)<br>

**Other Properties Related to Input:** The native reference inference profile is:

| Property | Setting |
| --- | --- |
| Temporal blocks | BEFORE, TARGET, AFTER, in this order; unavailable context is omitted |
| TARGET sampling | Duration-adaptive sampling at nominal 8 Hz, with 8–40 frames |
| Context sampling | Nominal 8 Hz, with 4–12 frames per available block |
| Total frame budget | At most 64 frames across all blocks |
| Frame preparation | Preserve chronological order and the full duration of each block when applying its frame budget; use temporal-pair-compatible frame counts |
| Spatial preparation | Preserve aspect ratio; cap each single-camera frame at 262,144 pixels, or each three-camera mosaic at 524,288 pixels, before the model processor's resize |
| Processor | Use the trained Qwen3.5 processor and chat template, with a 32,768-token sequence budget; the native processor uses `min_pixels=1024` and `max_pixels=589824` |

8 Hz is the requested sampling rate, not the recording's source frame rate. Duration caps and source-frame availability can change the effective sampling rate. Preserve each block's source clock and selected source-frame indices in the processor's video metadata; disable additional frame sampling with `do_sample_frames=False`. Short source spans require the reference frame-planning policy rather than invented observations. Missing context can be omitted; source-limited targets must be handled explicitly.

### Target intervals and camera layout

The caller supplies a half-open target interval `[start_s, end_s)` in the source video's clock. BEFORE and AFTER are the nearest available non-overlapping event intervals on either side, not a fixed number of seconds of padding. After temporal stitching, refresh these context intervals from the reconstructed event timeline. Sample all three blocks independently and retain their block-local source-frame indices and source frame rate. The captioner does not predict or revise the target boundaries.

For three-camera input, tile synchronized views into one RGB mosaic per sampled source time: the main camera occupies the upper panel, the left-wrist camera the lower-left panel, and the right-wrist camera the lower-right panel. Keep this arrangement identical across time and across all temporal blocks, and identify these roles in the prompt. Single-camera input uses ordinary RGB frames. Temporal blocks and camera panels represent different dimensions of the input: BEFORE/TARGET/AFTER select time, while the panels show simultaneous views.

### Example prompt

The `<video>` markers below stand for the trained processor's video tokens, with the corresponding ordered frame blocks attached. Omit a context block when unavailable.

```text
Each frame is a synchronized camera mosaic: main camera above, left wrist camera below left,
and right wrist camera below right. All panels show the same physical time.
Describe each physical operation once across the views.

BEFORE:
<video>

TARGET:
<video>

AFTER:
<video>

Write one concise, self-contained English instruction for the local operation being attempted in TARGET.
Use a natural imperative, concrete objects and the working hand; include meaningful support by the
other hand and necessary source or destination. Use BEFORE and AFTER only to disambiguate TARGET,
not to add operations occurring outside it. Do not invent an object, hand role or hidden goal.
Output only the instruction, on one line, without timestamps, execution-quality commentary or block names.
```

Omit the camera-mosaic paragraph for single-camera input. A failed or repeated attempt retains its established intended operation; the output should not turn an accident into a goal or assert an unobserved success.

## Output(s)
**Output Type(s):** Text<br>
**Output Format:** String<br>
**Output Parameters:** One-Dimensional (1D)<br>
**Other Properties Related to Output:**
* Flora-captioner-27B: One concise, self-contained English imperative describing the intended local operation in TARGET, with relevant objects and hand roles. No timestamps, context-only operations, or execution-quality commentary are requested.

**Illustrative output:**

```text
Place the blue mug on the tray with your right hand.
```

The data engine attaches this text to the supplied source-time interval. The timestamped annotation is a pipeline result, not a separate captioner output format.

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
* Flora-captioner-27B (v1): Textual annotation; fine-tuned from Qwen3.5-27B with supervised fine-tuning followed by reinforcement learning. It integrates into an AI system as the captioning stage of the Flora data-annotation pipeline: given a video and a target interval (for example, a segment produced by Flora-segmenter-9B), it returns one English action instruction, which the pipeline attaches to that interval in the episode annotation.

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
Robot camera video from multiple embodiments and human egocentric video of hand-object manipulation, paired with natural-language descriptions. Less than 10 million training examples, covering less than 10,000 hours of video. All training text is in English.

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

** Properties:
* WGO-Bench: The paper evaluates 100 human and robot manipulation episodes (25 HomER, 50 DROID, 25 Galaxea), containing 743 gold segments. Captioning is evaluated on gold intervals with the episode instruction supplied. The reported score is the equal-weight average across the three subsets.
* Modalities: Video and text. Linguistic characteristics: English task instructions and English reference captions.

**Benchmark Scores (reported in the Flora paper, Table 2):**

| Model | Benchmark | Metric | Flora |
| --- | --- | --- | --- |
| Flora-captioner-27B | WGO-Bench | Gold-window captioning accuracy, subset average (%) | 64.7 |

## Inference:
**Acceleration Engine:** PyTorch on CUDA (Hugging Face Transformers), using Brain Float 16 (BF16) with 32-bit floating point (FP32) Gated DeltaNet parameters
**Reference Decoding:** Use the trained processor and chat template with thinking disabled, greedy decoding, and at most 512 new tokens per target instruction. Supply the prepared frame blocks and their timing metadata without an additional processor sampling pass.
**Reference Precision:** Native inference preserves FP32 Gated DeltaNet parameters while using BF16 base weights. Preserve these mixed parameter dtypes when loading the weights.
**Test Hardware:**
* NVIDIA Hopper H100

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included. <br>

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
