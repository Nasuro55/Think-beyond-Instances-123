# Think beyond Instances

Official implementation of **Think beyond Instances**.

## 1. Installation

```bash
conda create -n TBI python=3.10 -y
conda activate TBI
pip install -r requirements.txt
```

## 2. Data Preparation

Install dataset dependencies:

```bash
pip install -r requirements-data.txt
```

Download the datasets:

```bash
python prepare_data.py --dataset math500
python prepare_data.py --dataset amc23
python prepare_data.py --dataset aime25
```

## 3. Run Experiments

Configure your model API in `configs/paper.json`, including `model_id` and `base_url`.

Run a single example:

```bash
python run.py --config configs/paper.json --question "If 3*x+7=22, what is x?"
```

Run evaluation on MATH500:

```bash
python evaluate.py --config configs/paper.json --dataset math500 --output results/math500
```

Run the enhanced version:

```bash
python evaluate.py --config configs/enhanced.json --dataset math500 --output results/math500_enhanced
```

Evaluation results are saved in `results/`.
