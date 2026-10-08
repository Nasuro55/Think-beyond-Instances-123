# Think beyond Instances

Python implementation of **Think beyond Instances (TBI)**, a multi-stage reasoning framework that incorporates expert reasoning, problem variation, answer verification, and rollback mechanisms to improve mathematical problem-solving.

## 1. Requirements

Python 3.10 is recommended.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the offline demo:

```bash
python demo.py
```

Run unit tests:

```bash
python -m unittest discover -s tests -t . -v
```

## 2. Configuration

Three configurations are provided:

| Configuration | Description |
|---|---|
| `configs/paper.json` | Core TBI framework with expert reasoning, problem variation, and rollback |
| `configs/enhanced.json` | Enhanced framework with additional verification and reliability mechanisms |
| `configs/local_8gb.json` | Local Qwen3-4B inference with NF4 quantization for 8GB CUDA GPUs |

### API-based Inference

The framework supports OpenAI-compatible Chat Completions APIs.

Configure the model in the corresponding configuration file:

```json
{
  "backend": "api",
  "model_id": "Qwen/Qwen3.5-9B",
  "base_url": "http://localhost:8000/v1",
  "api_key_env": "MODEL_API_KEY"
}
```

Set the API key when required:

```powershell
$env:MODEL_API_KEY = "YOUR_API_KEY"
```

Run inference:

```bash
python run.py --config configs/paper.json --question "If 3*x+7=22, what is x?"
```

For the enhanced framework:

```bash
python run.py --config configs/enhanced.json --question "If 3*x+7=22, what is x?"
```

### Local Model Inference

Install local inference dependencies:

```bash
pip install -r requirements-local.txt
```

Run Qwen3-4B with NF4 quantization:

```bash
python run.py --config configs/local_8gb.json --question "If 3*x+7=22, what is x?"
```

## 3. Data Preparation

The framework supports six mathematical reasoning benchmarks.

| Dataset | Source | Repetitions |
|---|---|---|
| MATH500 | `HuggingFaceH4/MATH-500` | 3 |
| Minerva Math | `math-ai/minervamath` | 3 |
| OlympiadBench | `Hothan/OlympiadBench` | 3 |
| College Math | `realtreetune/college_math` | 3 |
| AMC23 | `math-ai/amc23` | 32 |
| AIME25 | `math-ai/aime25` | 32 |

Dataset configurations are defined in `configs/datasets.json`.

Install dataset dependencies:

```bash
pip install -r requirements-data.txt
```

Download and prepare datasets:

```bash
python prepare_data.py --dataset math500
```

```bash
python prepare_data.py --dataset amc23
```

```bash
python prepare_data.py --dataset aime25
```

Custom datasets should use JSONL format:

```json
{"id": "1", "question": "If 3*x+7=22, what is x?", "answer": "5"}
```

## 4. Evaluation

### Quick Evaluation

Run a small-scale evaluation:

```bash
python evaluate.py --config configs/paper.json --dataset math500 --limit 5 --repeats 1 --output results/service_check
```

### Baseline

```bash
python evaluate.py --config configs/paper.json --dataset math500 --method baseline --output results/math500_baseline
```

### TBI Framework

```bash
python evaluate.py --config configs/paper.json --dataset math500 --output results/math500_paper
```

### Enhanced Framework

```bash
python evaluate.py --config configs/enhanced.json --dataset math500 --output results/math500_enhanced
```

### Performance Comparison

```bash
python compare_runs.py --reference results/math500_paper --candidate results/math500_enhanced --output results/comparison.json
```

### Evaluation Metrics

The evaluation pipeline reports:

- **Accuracy:** Overall prediction accuracy.
- **avg@N:** Average accuracy across repeated evaluations.
- **any_success@N:** Fraction of problems solved at least once.
- **Token Usage:** Input and output token consumption.
- **API Calls:** Number of model requests.
- **Failure Rate:** Proportion of unsuccessful executions.

Three answer-scoring methods are supported:

| Scorer | Description |
|---|---|
| `exact` | Exact string matching |
| `numeric` | Numerical equivalence |
| `math_verify` | Mathematical equivalence verification |

Install the optional mathematical verification dependencies:

```bash
pip install -r requirements-eval.txt
```

Evaluate using mathematical equivalence:

```bash
python evaluate.py --config configs/paper.json --dataset math500 --scorer math_verify --output results/math500_math_verify
```

## 5. Output Files

Each evaluation produces the following files:

```text
results/
├── manifest.json
├── predictions.jsonl
├── metrics.json
└── traces/
```

- `manifest.json`: Experimental configurations and metadata.
- `predictions.jsonl`: Predictions and evaluation results.
- `metrics.json`: Performance metrics and computational statistics.
- `traces/`: Intermediate reasoning, verification, and rollback records.

## 6. Project Structure

```text
ThinkBeyondInstances/
├── configs/               # Experiment configurations
├── data/                  # Dataset files
├── tbi/                   # Core framework
│   ├── backends.py        # Model interfaces
│   ├── prompts.py         # Reasoning prompts
│   ├── pipeline.py        # Main TBI framework
│   ├── arithmetic.py      # Arithmetic verification
│   ├── data.py            # Dataset processing
│   ├── evaluation.py      # Evaluation pipeline
│   └── scoring.py         # Answer scoring
├── tests/                 # Unit tests
├── run.py                 # Single-problem inference
├── evaluate.py            # Benchmark evaluation
├── prepare_data.py        # Dataset preparation
├── compare_runs.py        # Performance comparison
├── demo.py                # Offline demonstration
├── requirements.txt
├── requirements-local.txt
├── requirements-data.txt
└── requirements-eval.txt
```
