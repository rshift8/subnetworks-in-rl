# Ablation Study for Subnetworks Emerging from RL Finetuning

An experiment designed as part of the [MAT1510](https://sites.google.com/view/mat1510/fall-2025?authuser=0) final project at the University of Toronto by Richard Shi. See the [project report](report/report.pdf) for the paper summary, experimental design, results, and discussion.

## Overview

This project investigates how sensitive a language model's mathematical reasoning outputs are to changes in a subnetwork identified from differences between supervised-finetuned (SFT) and reinforcement-learning-finetuned (RL) checkpoints. It is motivated by Mukherjee et al., *Reinforcement Learning Finetunes Small Subnetworks in Large Language Models* ([arXiv:2505.11711](https://arxiv.org/abs/2505.11711)). As summarized in the report, that paper finds sparse RL parameter updates and explores training only the affected subnetwork.

This repository performs inference-time ablations using existing checkpoints; it does not train models with SFT or RL.

## Experimental Design

The study uses three publicly available checkpoints:

| Role in the experiment | Checkpoint |
| --- | --- |
| Base model | `Qwen/Qwen2.5-Math-1.5B` |
| SFT comparison model | `VerlTool/Qwen2.5-Math-1.5B-TIR-SFT` |
| Instruct model, used as the RL checkpoint | `Qwen/Qwen2.5-Math-1.5B-Instruct` |

The report assumes that the Instruct checkpoint approximates the result of RL finetuning the SFT checkpoint. These checkpoints come from different sources, and that training relationship is **not established**, which limits conclusions about RL-specific changes.

The report defines the subnetwork as parameters satisfying `abs(theta_RL - theta_SFT) > 1e-2`. The threshold exploration in `subnetworks.ipynb` found that approximately **85.75%** of parameter differences were within `1e-2` of zero, motivating that tolerance.

Starting from the Instruct model, the intended ablations reset weights to their values in the **base model**:

1. Reset weights inside the identified subnetwork.
2. Reset weights outside the identified subnetwork.
3. Reset a random set of weights with the same number of selected parameters per layer as the subnetwork (seed `42`).

The three altered models are compared with the base and Instruct models using three prompts: a linear equation (`4x + 5 = 6x + 7`), a three-variable linear system, and an indefinite integral (`integral of w^(1/3) + 10w^(3/5)`). Outputs are inspected qualitatively, with up to `4096` new tokens per response.

## Findings and Limitations

The report describes mixed results. On the first prompt, models produced the correct answer, but altered models added repetitive Markdown blocks. On the second, similar formatting issues appeared, and the Instruct model became stuck in an erroneous reasoning loop. On the integral, the report describes correct, broadly similar answers from the altered models. Overall, the altered models behaved similarly enough that this small experiment did not establish a clear difference in sensitivity to subnetwork versus other weight changes.

These observations come from only three hand-inspected prompts, without benchmark scores or repeated trials. The unmatched checkpoint histories and implementation discrepancies below further limit the interpretation. The findings should be treated as exploratory rather than confirmation of the source paper's results.

## Repository Contents

| File | Purpose |
| --- | --- |
| [subnetworks.ipynb](subnetworks.ipynb) | Explores parameter differences and tolerances to identify a threshold with high update sparsity. |
| [ablation.ipynb](ablation.ipynb) | Main experiment: constructs masks, saves altered models, and generates responses after resetting selected weights to base-model values. Also contains the earlier zero-weight ablations. |
| [evaluation.ipynb](evaluation.ipynb) | Historical notebook for experiments with zeroed weights. Those results were uninsightful and the approach was discarded; the notebook remains for reference and contains unresolved variable/name issues. |
| [results/responses_simpler_base_mask_maxlength4096.json](results/responses_simpler_base_mask_maxlength4096.json) | Saved responses for the base-weight ablations, organized by prompt and model. |
| [report/report.pdf](report/report.pdf) | Final project report. |
| [requirements.txt](requirements.txt) | Experiment dependencies: PyTorch, Transformers, NumPy, and tqdm. |

## Setup

From the repository root, create and activate a Python virtual environment, then install the experiment dependencies and a notebook interface:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name subnetworks-in-rl --display-name "Subnetworks in RL"
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. You can also use an existing Python environment and install the same dependencies there.

The first model load downloads checkpoints from Hugging Face, so internet access and disk space for the downloads are required. Generated model directories also occupy several GB each. The notebooks hold multiple 1.5B-parameter models and large intermediate tensors in memory, so allow substantial system RAM. Evaluation uses CUDA when available and otherwise falls back to CPU; model loading and mask construction run on CPU as written. The SFT and Instruct loads use `torch.bfloat16`.

Dependencies and checkpoint revisions are not pinned, so the repository does not define an exact reproducible software environment.

## Execution

Launch JupyterLab from the repository root and select the **Subnetworks in RL** kernel:

```bash
jupyter lab
```

You can also open the notebooks in VS Code and select the same environment. Keep the notebook working directory at the repository root so generated model paths resolve consistently.

### 1. Explore update sparsity (optional)

Run `subnetworks.ipynb` from top to bottom. It loads the SFT and Instruct checkpoints, computes their parameter differences, and prints the fraction near zero for tolerances from `1e-5` to `1e-2`. The main ablation notebook computes its own mask, so this step is not a prerequisite. Shut down this notebook's kernel afterward to release its model tensors before running the main experiment.

### 2. Review the implementation discrepancies

Before generating new models in `ablation.ipynb`, choose whether to inspect the historical implementation or align the experiment with the report. Two changes are needed for the report's intended design:

- In `compute_subnetwork_mask`, replace `delta_mask[name] = delta > tolerance` with `delta_mask[name] = delta.abs() > tolerance`. The checked-in code selects only positive changes above the threshold.
- In the call that creates `base_non_subnetwork`, pass `invert=True`:

  ```python
  base_mask(base_weights, subnetwork_mask, 'base_non_subnetwork', invert=True)
  ```

  The checked-in call omits this argument, so it resets the same selected weights as `base_subnetwork`. The saved responses for those two model names are also identical for all three prompts.

The integral prompt also uses ordinary Python strings containing `\frac`, where `\f` becomes a form-feed character. For a fresh run, replace the third prompt with a raw string:

```python
prompts[2] = r'\int (w^{\frac{1}{3}} + 10 w^{\frac{3}{5}})dw = '
```

These corrections change the experiment inputs or masks; new outputs should be treated as a new run rather than a reproduction of the committed responses.

### 3. Build the ablated models

In `ablation.ipynb`, run the **Imports**, **Loading Models**, and **Util Functions** sections in order. These create the subnetwork mask with tolerance `1e-2` and the random mask with seed `42`.

Skip **Models for Ablation Study (Zero Mask)** for the main experiment. That section creates the earlier `zero_subnetwork`, `zero_non_subnetwork`, and `random_subnetwork` models and is not needed by the base-weight evaluation.

Run **Models for Ablation Study (Base Model Mask)** next. It saves three model directories in the working directory:

```text
base_subnetwork/
base_non_subnetwork/
base_random_subnetwork_seed42/
```

These directories are generated locally and are not included in the repository. Run this section once per mask configuration; rerunning it writes to the same output directories.

### 4. Generate and inspect responses

Run the **Eval** section in order, applying the raw-string prompt correction above after the `prompts` cell if desired. The notebook loads each of the five models for each prompt and generates up to `4096` new tokens. This can take considerable time on CPU.

The evaluation function applies a chat template to every model, including the base checkpoint. If your installed tokenizer reports that the base checkpoint has no chat template, tokenize the base prompt directly with `tokenizer(prompt, return_tensors="pt").to(model.device)` in that branch, and retain `apply_chat_template` for the other models. This changes base-model prompting and should be recorded when comparing runs.

The final cell saves responses to:

```text
responses_simpler_base_mask_maxlength4096.json
```

This output is written in the notebook's working directory. The committed historical output is under `results/`; change `fp` in the final cell if you want a different destination. Each JSON entry maps a prompt to model names and lists of generated responses. No automatic correctness scoring is performed.

To inspect the existing results without downloading or running models:

```bash
python3 -m json.tool results/responses_simpler_base_mask_maxlength4096.json
```
