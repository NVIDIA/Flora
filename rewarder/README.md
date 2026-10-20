---
base_model: Qwen/Qwen3.5-27B
library_name: transformers
pipeline_tag: video-text-to-text
tags:
  - robotics
  - reward-model
  - vlm
  - gr00t
  - system-2
---

# Flora-rewarder-27B

# Model Overview

### Description:

Flora-rewarder-27B produces outcome rewards (an episode-level success score), used for monitoring real-world deployments, and process rewards (dense action-quality scores over short time steps), used during reinforcement learning-based post-training. Flora was developed by NVIDIA as the data engine used for training GR00T-2, and is released to help researchers scale up custom robot datasets.

This model is for research and development only.

### License/Terms of Use:
Use of this model is governed by the [NVIDIA OneWay Noncommercial License (22 March 2022)](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf).<br>
This model has been post-trained on `Qwen/Qwen3.5-27B`, revision `fc05daec18b0a78c049392ed2e771dde82bdf654`. The Qwen base model is licensed under [Apache License 2.0](https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/LICENSE).<br>
You are responsible for ensuring that your use of NVIDIA provided models complies with all applicable laws. <br>

### Deployment Geography:
Global

### Use Case:
Researchers, Academics, Open-Source Community: Annotating robot datasets with rewards; reward modeling for robot reinforcement learning; and research into emerging applications such as physical auto-research and human-to-robot transfer.<br>
Developers: Prototyping and evaluating data curation, filtering, and monitoring pipelines for robotics foundation model research.<br>

### Release Date:
- **Hugging Face:** 10/15/2026 via [Flora-Rewarder](https://huggingface.co/nvidia/Flora-Rewarder).

### Intended Use

Flora is intended for non-commercial robotics research and development only, including producing reward signals for policy evaluation and reinforcement learning, and monitoring robot deployments in research settings. Flora is not intended or supported for commercial or production use. Flora outputs are annotations. It is not intended to directly control robots or to serve as the sole safety monitor for a physical system.

## References(s):
* GR00T Website: [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/gr00t)
* GR00T N1 White Paper: https://arxiv.org/abs/2503.14734
* Base Model: [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)<br>
* Benchmarks: [RoboReward](https://huggingface.co/datasets/teetone/RoboReward), [RoboMeter](https://robometer.github.io/)<br>

## Model Architecture:
**Architecture Type:** Transformer

**Network Architecture:** Other: Qwen3.5 vision-language model (natively multimodal vision encoder with a decoder-only Large Language Model)

**This model was developed based on Qwen3.5-27B.**

**Number of model parameters:**
* Flora-rewarder-27B: 2.7 x 10^10

Flora-rewarder-27B was trained with full fine-tuning of all parameters, including the vision encoder. Sampled video frames from one or more camera views are encoded by the vision encoder. The resulting visual tokens are concatenated with text tokens from the prompt, which specifies the annotation task and optionally the task instruction, and passed to the language model decoder. The decoder autoregressively generates the annotation as text. The rewarder emits structured text that parses into an outcome score, dense action-quality scores, or rubric step scores. The reward model can output multiple different types of rewards.



## Input(s):
**Input Type(s):**
* Video: Robot or human egocentric camera frames<br>
* Text: Prompt specifying the annotation task, optionally with the task instruction<br>

**Input Format(s):**
* Video: Red, Green, Blue (RGB) frames (e.g., .mp4)<br>
* Text: String<br>

**Input Parameters:**
* Video: Four-Dimensional (4D): a sequence of RGB images (frames × height × width × 3)<br>
* Text: One-Dimensional (1D)<br>

**Other Properties Related to Input:** One or more camera views per episode. Outcome and dense prompts receive a single video in which each sampled time is a mosaic of up to four camera views. Outcome rewards use up to 64 frames sampled uniformly across the episode (about 10 frames per second of video, at least 2). Dense rewards use 16-second windows of 64 frames sampled at 4 frames per second (FPS). Rubric prompts receive each camera as a separate video sampled at 1 FPS, with at most 82 frames per camera, each resized to between 65,536 and 131,072 pixels. Outcome and dense prompts are at most 16,384 tokens; rubric prompts are at most 32,768 tokens.

## Output(s)
**Output Type(s):** Text<br>
**Output Format:** String<br>
**Output Parameters:** One-Dimensional (1D)<br>
**Other Properties Related to Output:**
* Outcome reward: An integer episode score from 1 (complete failure) to 5 (complete success), optionally preceded by a rationale.
* Dense reward: For a 16-second window, an action-quality score for each 0.5-second step (1 = failure, 2 = suboptimal, 3 = normal), the worst score in the window, and rationales for non-normal steps.
* Rubric reward: A reasoning trace followed by one score per rubric step.

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
* Flora-rewarder-27B (v1): Outcome and process rewards; fine-tuned from Qwen3.5-27B. It integrates into an AI system as a reward model in robot learning and evaluation pipelines: the pipeline sends recorded episodes or rollout video windows with the task instruction, parses the returned text into outcome, dense, or rubric scores, and uses those scores for policy evaluation, data filtering, deployment monitoring, or reinforcement learning.

## Training, Testing, and Evaluation Datasets:

### Training Data Summary

Training labels are a combination of human-written annotations, automatic success signals recorded with the data, and VLM-generated annotations.

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
Robot camera video from multiple embodiments, paired with reward labels. Less than 1 million training examples spanning outcome, dense, and rubric-derived rewards, covering less than 10,000 hours of video. All training text is in English.

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
* RoboRewardBench: Episodes with human-labeled outcome scores on a 1–5 scale.
* RobometerBench: Policy ranking on 567 trajectories from six out-of-distribution datasets. Kendall’s tau-a is averaged over tasks within each dataset and then equally across datasets.
* Modalities: Video and text. Linguistic characteristics: English task instructions; labels are numeric outcome scores and policy rankings.

**Benchmark Scores (reported in the Flora paper, Table 4 and Appendix F):**

| Model | Benchmark | Metric | Flora |
| --- | --- | --- | --- |
| Flora-rewarder-27B | RoboRewardBench | MAE on the 1–5 outcome scale, multiplied by 100 | 40.0 |
| Flora-rewarder-27B | RobometerBench, OOD policy ranking | Kendall’s tau-a, multiplied by 100 | 72.0 |

## Inference:
**Acceleration Engine:** PyTorch on CUDA (Hugging Face Transformers), using Brain Float 16 (BF16)
**Test Hardware:**
* NVIDIA Hopper H100

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included. <br>

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
