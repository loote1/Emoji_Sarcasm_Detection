# Emoji Sarcasm Detection

中文 emoji 反讽检测实验项目。项目包含数据清洗与划分脚本、一个中文适配的 ESD-ZH 模型（emoji + 文本混合 token、BiGRU + Attention），以及 Random、TF-IDF + LR、RoBERTa、Ollama/DeepSeek LLM 等 baseline。

## 目录结构

```text
.
├── train_esd_zh.py                  # ESD-ZH 主模型训练与评估
├── final_args.md                    # 当前主模型推荐/记录参数
├── data/
│   ├── emoji_comments.csv           # 原始数据
│   ├── emoji_train.csv              # 训练集
│   ├── emoji_val.csv                # 验证集
│   └── emoji_test.csv               # 测试集
├── scripts/
│   ├── process_data.py              # 清洗原始 CSV
│   ├── split_emoji_dataset.py       # 划分 train/val/test
│   └── summarize_baselines.py       # 汇总 artifacts 下的 metrics
├── baseline_scripts/
│   ├── baseline_random.py
│   ├── baseline_tfidf_lr.py
│   ├── baseline_roberta.py
│   ├── baseline_llm_ollama.py
│   └── baseline_llm_deepseek.py
├── prompts/
│   └── prompts.txt                  # LLM baseline 提示词文件
├── model_visual_frontend/            # ESD-ZH 模型可视化前端与推理服务
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   ├── server.py                     # 本地 HTTP 服务与 /predict 推理接口
│   └── requirements.txt              # 前端推理服务依赖
└── artifacts/                       # 训练结果、模型权重和指标
```

## 环境准备

建议使用 Python 3.10+。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn regex tqdm torch transformers requests
```

如果要运行 RoBERTa baseline，首次执行会从 Hugging Face 下载模型 `hfl/chinese-roberta-wwm-ext`。如果要使用 GPU，请先根据你的 CUDA 版本安装对应的 PyTorch。

## 前端可视化

`model_visual_frontend/` 提供一个本地可视化页面，用于输入中文短文本并调用已训练的 ESD-ZH 模型进行反讽概率预测。默认模型文件为 `artifacts/esd_zh/model.pt`。

安装前端推理服务依赖：

```powershell
pip install -r model_visual_frontend/requirements.txt
```

启动本地服务：

```powershell
python model_visual_frontend/server.py --host 127.0.0.1 --port 8080 --device cpu
```

启动后在浏览器打开：

```text
http://127.0.0.1:8080/
```

服务同时提供接口：

- `GET /health`：检查服务和模型状态
- `POST /predict`：提交 `{"text": "待检测文本"}`，返回预测标签、反讽概率、阈值和分词结果

如果要使用 GPU 推理，可以在已正确安装 CUDA 版 PyTorch 后将 `--device cpu` 改为 `--device cuda`。

## 数据格式

训练、验证、测试 CSV 默认使用两列：

| 列名 | 含义 |
| --- | --- |
| `text` | 中文短文本，允许包含 emoji |
| `irony` | 标签，`1` 表示反讽/讽刺，`0` 表示非反讽 |

示例：

```csv
text,irony
呜呜妈妈果然是世界上最爱你的人🥺🥺,0
真的可以拿上国际舞台吗🤔️,0
```

## 数据清洗与划分

如果已有 `data/emoji_train.csv`、`data/emoji_val.csv`、`data/emoji_test.csv`，可以跳过本节。

从原始 CSV 提取文本列与标签列：

```powershell
python scripts/process_data.py `
  --input data/emoji_comments.csv `
  --output data/emoji_comments.cleaned.csv
```

脚本会尝试识别 `comment`/`text`/`content` 作为文本列，识别 `irony`/`label`/`y` 作为标签列，并输出标准列名 `text`、`irony`。

划分训练集、验证集和测试集：

```powershell
python scripts/split_emoji_dataset.py `
  --input data/emoji_comments.cleaned.csv `
  --text_col text `
  --label_col irony `
  --train_ratio 0.8 `
  --val_ratio 0.1 `
  --test_ratio 0.1 `
  --dedup_text
```

输出文件：

- `data/emoji_train.csv`
- `data/emoji_val.csv`
- `data/emoji_test.csv`
- `data/emoji_split_stats.json`

## 训练 ESD-ZH 主模型

默认训练：

```powershell
python train_esd_zh.py
```

使用当前记录参数训练：

```powershell
python train_esd_zh.py `
  --train_csv data/emoji_train.csv `
  --val_csv data/emoji_val.csv `
  --test_csv data/emoji_test.csv `
  --text_col text `
  --label_col irony `
  --emb_dim 300 `
  --hidden_dim 64 `
  --attn_dim 64 `
  --max_len 160 `
  --lr 0.002 `
  --dropout 0.5 `
  --batch_size 64 `
  --min_freq 1 `
  --splits both `
  --output_dir artifacts/esd_zh
```

训练输出：

- `artifacts/esd_zh/model.pt`：模型权重、词表和参数
- `artifacts/esd_zh/vocab.json`：token 词表
- `artifacts/esd_zh/metrics.json`：训练历史与验证/测试指标

`train_esd_zh.py` 会自动使用 CUDA；如需强制 CPU：

```powershell
python train_esd_zh.py --device cpu --splits both
```

## 运行 Baseline

随机/多数类 baseline：

```powershell
python baseline_scripts/baseline_random.py --splits both
```

TF-IDF + Logistic Regression：

```powershell
python baseline_scripts/baseline_tfidf_lr.py --splits both
```

RoBERTa 微调：

```powershell
python baseline_scripts/baseline_roberta.py `
  --model_name hfl/chinese-roberta-wwm-ext `
  --splits both
```

Ollama 本地大模型 baseline：

```powershell
ollama serve
ollama pull qwen2.5:7b-instruct
python baseline_scripts/baseline_llm_ollama.py `
  --model qwen2.5:7b-instruct `
  --prompt_file prompts/prompts.txt `
  --splits both
```

如果 `prompts/prompts.txt` 打开后是乱码，请先替换成你自己的 UTF-8 中文提示词文件，再传给 `--prompt_file`。

DeepSeek API baseline：

```powershell
$env:DEEPSEEK_API_KEY="你的 API Key"
python baseline_scripts/baseline_llm_deepseek.py `
  --model deepseek-chat `
  --prompt_file prompts/prompts.txt `
  --splits both
```

快速试跑 LLM baseline 时可以限制样本数：

```powershell
python baseline_scripts/baseline_llm_ollama.py `
  --model qwen2.5:7b-instruct `
  --prompt_file prompts/prompts.txt `
  --splits val `
  --max_examples 20
```

## 汇总实验结果

所有脚本都会把指标保存为 `metrics.json`。统一汇总：

```powershell
python scripts/summarize_baselines.py `
  --artifacts_dir artifacts `
  --output_csv artifacts/summary.csv `
  --output_json artifacts/summary.json
```

当前项目中已有汇总结果：

- `artifacts/summary.csv`
- `artifacts/summary.json`

## 常用参数说明

ESD-ZH 主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--max_len` | `160` | 单条文本最大 token 长度 |
| `--min_freq` | `1` | 构建词表时的最低词频 |
| `--max_vocab` | `50000` | 最大词表大小 |
| `--emb_dim` | `300` | embedding 维度 |
| `--hidden_dim` | `64` | BiGRU 隐层维度 |
| `--attn_dim` | `64` | attention 中间维度 |
| `--dropout` | `0.5` | dropout |
| `--batch_size` | `64` | batch size |
| `--lr` | `0.002` | 学习率 |
| `--epochs` | `50` | 最大训练轮数 |
| `--patience` | `5` | early stopping 容忍轮数 |
| `--splits` | `val` | 评估 `val`、`test` 或 `both` |

## 注意事项

- 标签必须是二分类整数 `0`/`1`。
- CSV 默认按 `utf-8-sig` 读取，适合 Windows/Excel 导出的带 BOM 文件。
- LLM baseline 的默认内置 prompt 存在编码问题，`prompts/prompts.txt` 也可能已经是乱码；建议自行提供 UTF-8 prompt 文件并通过 `--prompt_file` 传入。
- RoBERTa 和 LLM baseline 运行时间较长；调参时建议先用 `--splits val` 或 `--max_examples` 小样本验证流程。
