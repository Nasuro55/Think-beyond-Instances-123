# Think beyond Instances — Python 实现

## 先选运行方式

| 配置 | 用途 | 特点 |
|---|---|---|
| `configs/paper.json` | 实现论文核心流程 | 同一个模型担任所有角色；每步 4 卡片；单个有效变题；整链最多回滚一次 |
| `configs/enhanced.json` | 尝试提高可靠性 | 两个有效变题、精确算术检查、专家一致时复核、框架答案复核及有限重试 |
| `configs/local_8gb.json` | 本机 8GB CUDA 显存 | Qwen3-4B、NF4 量化、串行专家、较小生成与输入预算 |

增强项是可选研究扩展，**尚未证明提高真实基准准确率**。先用独立开发集比较，再固定配置运行测试集。增强版多用了模型调用，比较时应同时报告准确率和成本。

## 不下载模型即可检查代码

在 PowerShell 中运行：

```powershell
cd "$env:USERPROFILE\Desktop\ThinkBeyondInstances"
python demo.py
python demo.py --enhanced --output results/offline_demo_enhanced.json
python -m unittest discover -s tests -t . -v
```

`demo.py` 是明确标注的**预设回复回放**，用于检查流程，既不调用真实模型，也不代表实验成绩。它演示论文示意题的三位专家分歧、逻辑卡片筛选及最终答案 `58`。没有安装可选评分依赖时，相关五项评分测试会跳过。

要求 Python 3.10 或以上。API 后端、普通 JSONL 评估、回放和基础测试只依赖标准库；不要安装桌面旧项目的大量 GPU 依赖来运行此项目。

## 方式 A：API 模型

适用于已经运行的本地模型服务，也适用于兼容 Chat Completions 格式的远程服务。项目默认填入：

```json
{
  "backend": "api",
  "model_id": "Qwen/Qwen3.5-9B",
  "model_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
  "base_url": "http://localhost:8000/v1",
  "api_key_env": "MODEL_API_KEY"
}
```

`model_url` 是模型权重来源；真正调用哪个模型由 `model_id` 决定。`base_url` 是你实际运行或购买的模型服务地址。默认的 localhost 地址**不会自动启动服务**。

远程服务可按供应商要求改成英文占位示例：

```json
{
  "model_id": "INSERT_PROVIDER_MODEL_NAME_HERE",
  "base_url": "https://INSERT_MODEL_SERVER_HOST_HERE/v1"
}
```

如需密钥，在当前终端设置环境变量：

```powershell
$env:MODEL_API_KEY = "INSERT_API_KEY_HERE"
python run.py --config configs/enhanced.json --question "Given real numbers x and y with x+y=10 and x**3+y**3=370, find x**2+y**2." --output results/my_problem.json
```

项目不把密钥写入配置或调用日志。本地服务不要求密钥时可不设置该变量。

如果服务不支持 `seed`，将 `send_seed` 改为 `false`。缓存仍会区分不同重复实验，但供应商是否可复现取决于其实现。若支持 JSON 模式，可设置 `json_mode: true`。Qwen 服务可按服务器文档在 `extra_body` 中加入 `chat_template_kwargs`，例如：

```json
{"chat_template_kwargs": {"enable_thinking": true}}
```

`max_tokens` 包含服务端可能计入的思考 token。若频繁出现 `finish_reason=length`，增大预算；代码会拒绝截断回复。默认预算是实现选择，论文没有提供完整的长度设置。

## 方式 B：本机 Qwen3-4B 量化

你的显卡为 RTX 4060 Laptop，约 8GB 显存。建议在单独环境中使用量化配置，不要直接加载 Qwen3.5-9B 全精度权重。

```powershell
cd "$env:USERPROFILE\Desktop\ThinkBeyondInstances"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe run.py --config configs/local_8gb.json --question "If 3*x+7=22, what is x?"
```

本地运行需要与你的系统匹配的 CUDA 版 PyTorch。可先检查：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

若为 `False`，按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装 CUDA 对应版本。bitsandbytes 的支持环境见 [官方安装说明](https://huggingface.co/docs/bitsandbytes/main/en/installation)。本地适配器只处理 Qwen3 等纯文本因果模型；Qwen3.5 请使用 API 后端。

首次执行本地命令会下载权重到项目内 `cache/model`。未替你下载大模型，也未测试本机真实模型推理。量化、短上下文和较小输出预算会影响结果，不能直接当作论文全精度实验。

## 数据集入口

`configs/datasets.json` 已填入六个基准的数据地址、子集、字段和重复次数：

| 名称 | Hugging Face 数据地址 | 默认重复次数 |
|---|---|---|
| MATH500 | `HuggingFaceH4/MATH-500` | 3 |
| Minerva Math | `math-ai/minervamath` | 3 |
| OlympiadBench | `Hothan/OlympiadBench` / `OE_TO_maths_en_COMP` | 3 |
| College Math | `realtreetune/college_math` | 3 |
| AMC23 | `math-ai/amc23` | 32 |
| AIME25 | `math-ai/aime25` | 32 |

OlympiadBench 所选的是英语、纯文本、开放题数学竞赛子集；上游把这些题放在 `train` 命名的分区，本项目只做推理，不使用它训练。College Math 使用可访问镜像。这两项以及 Minerva 的确切数据版本应在严格复现前与论文原始数据核对；不能声称默认入口就是作者的精确切分。

下载示例：

```powershell
python -m pip install -r requirements-data.txt
python prepare_data.py --dataset math500
python prepare_data.py --dataset amc23
python prepare_data.py --dataset aime25
```

下载工具会记录解析后的 Hugging Face commit revision。需要试运行时可用 `--limit 5 --output data/math500_small.jsonl`。已有目标文件不会被覆盖。下载失败不会偷偷换成示例题。

你也可以提供自己的英文占位路径：

```powershell
python evaluate.py --data "INSERT_DATASET_JSONL_PATH_HERE" --config configs/enhanced.json --repeats 3 --output results/custom_run
```

自定义 JSONL 每行一题，标准格式如下：

```json
{"id":"unique_problem_id","question":"INSERT_MATH_PROBLEM_HERE","answer":"INSERT_REFERENCE_ANSWER_HERE"}
```

`answer` 必须是最终答案，不能用整段 `solution` 替代；`problem` 可作为 `question` 的别名。参考答案只有评分模块会读取，**不会传给求解模块**。多个必须同时满足的答案保留为 JSON 列表字符串，不当作任选一个即可的候选答案；这类题应核对评分规则或使用专门适配器。

## 运行实验和比较

先用少量题检查实际服务、回复格式和 token 预算：

```powershell
python evaluate.py --config configs/paper.json --dataset math500 --limit 5 --repeats 1 --output results/service_check
```

正式比较时使用相同模型、题目、重复次数和评分口径：

```powershell
python evaluate.py --config configs/paper.json --dataset math500 --method baseline --output results/math500_baseline
python evaluate.py --config configs/paper.json --dataset math500 --output results/math500_paper
python evaluate.py --config configs/enhanced.json --dataset math500 --output results/math500_enhanced
python compare_runs.py --reference results/math500_paper --candidate results/math500_enhanced --output results/comparison.json
```

中断后使用原配置、原数据和原命令加 `--resume`。代码变化、配置变化或数据变化后应创建新的输出目录，避免把不一致的实验拼起来。已经记录的错误试次也视为已完成；要重做它们请另建实验目录。

每个实验保存：

- `manifest.json`：配置、数据散列、代码散列、重复次数、评分口径和数据来源。
- `predictions.jsonl`：每个试次的预测、实际评分、状态和调用成本。
- `traces/`：真实专家输出、卡片候选、变题、路由反馈、回滚和错误记录。
- `metrics.json`：总体正确率、各次运行正确率、avg@N、失败率及 token/调用统计。

AMC23 和 AIME25 默认做 32 次完整且独立的框架试次。`avg_at_n` 是这些试次正确率的平均值，**不是投票成绩，也不是任一次答对就算对**。`any_success_at_n` 单独列出，不能替代 avg@32。

`compare_runs.py` 给出按题目成组抽样的配对 bootstrap 区间，并报告两组实际成本；它没有把两种方法的计算预算假定为相等。此项目的 single-model baseline 不是论文表 1 的所有训练型基线，也不是论文的等预算 self-consistency 基线。

## 三种评分口径

- `--scorer exact`：默认，比较完整答案字符串，不做数值提取或模板归一化。它是明确的工程代理评分，论文没有给出可执行的官方 grader，不能直接称为作者原始评分协议。
- `--scorer numeric`：只比较完整的有理数标量表达式，例如 `1/2` 与 `0.5`。不从解释中抽取最后一个数字，不去掉百分号，不把坐标括号抹掉。
- `--scorer math_verify`：可选标准数学等价评分，支持更多公式和 LaTeX。它会改变评分协议；应单独标注，并在 baseline 和 TBI 中使用同一口径。

安装并使用可选评分器：

```powershell
python -m pip install -r requirements-eval.txt
python evaluate.py --config configs/enhanced.json --dataset math500 --scorer math_verify --output results/math500_enhanced_math_verify
```

Math-Verify 的参考实现见 [官方仓库](https://github.com/huggingface/Math-Verify)。本项目只评估结构化的最终 `answer` 字段；评分在独立进程中运行，设置 12 秒硬期限。额外保留纯数值百分比的含义，并拒绝把集合与有序坐标视作同类。评分超时、解析失败或错误会写入记录并算作未通过；多答案和不常见表示仍需人工核对。

## 文件结构

```text
ThinkBeyondInstances/
  run.py                 Single-problem inference
  evaluate.py            Benchmark evaluation and resume
  prepare_data.py        Benchmark download and normalization
  compare_runs.py        Paired accuracy/cost comparison
  demo.py                Clearly labeled offline replay
  configs/               Model, method, and dataset configuration
  data/example.jsonl     Small illustrative problems
  tbi/config.py          Configuration validation and seeds
  tbi/backends.py        API and local Transformers adapters
  tbi/prompts.py         English role prompts and JSON schemas
  tbi/schemas.py         Strict response parsing
  tbi/pipeline.py        Main three-stage method
  tbi/arithmetic.py      Conservative exact arithmetic checking
  tbi/data.py            Dataset adapters
  tbi/evaluation.py      Label-isolated evaluation
  tbi/scoring.py         Optional grader launcher
  tbi/scoring_worker.py  Isolated mathematical equivalence grader
  tests/                 Control-flow, HTTP, and grading tests
```

实现对照与未验证项见 `IMPLEMENTATION_NOTES.md`。接口地址和凭据尚需连接到你的真实模型服务；没有捏造基准成绩。
