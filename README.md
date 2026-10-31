# MP2 — How to Verify and Reproduce Results

This README is edited from [the one in the original repository](https://github.com/xudongwu-0/DASE7506-MP2/blob/main/code/README.md).

## 1. Install and check the data

Clone the complete repository and use **Python 3.12**:

```bash
git clone https://github.com/naming-is-painful/DASE7506-MP2.git
cd DASE7506-MP2/code
python -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy Unrestricted) ; (& .\.venv\Scripts\Activate.ps1)
python -m pip install -r requirements.txt
python check_data.py
```

Alternatively, use **Code → Download ZIP** on [GitHub](https://github.com/naming-is-painful/DASE7506-MP2), extract it, enter its `code/` directory and run the environment/install commands above.

You may need to manually confirm running the script in PowerShell.

## 2. Result Validation

The following commands verify the results:

```bash
$env:OMP_NUM_THREADS=4
$env:OPENBLAS_NUM_THREADS=4
$env:MKL_NUM_THREADS=4
python predict.py --method submission.py --checkpoint runs/model_bundle --split test --output runs/test-predictions.json
python score_predictions.py --split test --predictions runs/test-predictions.json --output runs/test-score.json
```

Run the following commands to reproduce the model parameters:

```bash
$env:OMP_NUM_THREADS=4
$env:OPENBLAS_NUM_THREADS=4
$env:MKL_NUM_THREADS=4
python train.py
python predict.py --method submission.py --checkpoint runs/model_bundle --split validation --output runs/validation-predictions.json
python score_predictions.py --split validation --predictions runs/validation-predictions.json --output runs/validation-score.json
```

## 3. Data and files

| Files                                 | Purpose                                                                            |
| ------------------------------------- | ---------------------------------------------------------------------------------- |
| `submission.py`                       | Empty prediction interface for your implementation.                                |
| `predict.py`                          | Runs your frozen method; writes predictions without reading references.            |
| `score_predictions.py`, `common.py`   | Fixed independent scorer, numeric parser and data loading.                         |
| `check_data.py`                       | Checks hashes, split disjointness and feature/reference alignment.                 |
| `data/candidates/gsm8k_{split}.jsonl` | Questions and four candidate traces.                                               |
| `data/candidates/gsm8k_{split}.npz`   | Fixed whole-trace embeddings; no model weights or correctness labels.              |
| `data/gsm8k_{split}.jsonl`            | Aligned references: training supervision, validation selection, test scoring only. |
| `data/manifest.json`                  | Protocol, counts, provenance and data hashes.                                      |
| `data/LICENSE.txt`, `data/NOTICE.txt` | Source license and attribution.                                                    |
| `PACKAGE_MANIFEST.json`               | Release file hashes, relative to the package root containing code/ and guide/.     |

Protocol: **`dase7506-mp2-open-hard-verifier-v5`**. Splits contain **4,755 / 798 / 467** train/validation/test questions and do not overlap. Every group has a correct final answer. Training and validation include easier groups; test questions have at least two different valid parsed answers. Candidate order is fixed. Test references are public for self-reporting and peer reproduction; they must not enter training, lookup tables or method selection.

## 4. Measurement and reproduction

The model is trained on my laptop with `AMD Ryzen 7 7840H` CPU. I am unsure about precisely measuring prediction time, but it is well under 1s.

The `/runs` folder is 133 KiB, and the `/runs/model_bundle` folder is less than 26 KiB.

## 5. Attribution

Questions and references derive from [GSM8K](https://github.com/openai/grade-school-math), distributed under its retained MIT license. Candidate solutions and fixed representations were prepared with [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B); the model revision is recorded in the data manifest. No model weights are distributed. Keep `data/LICENSE.txt` and `data/NOTICE.txt` when redistributing this package.
