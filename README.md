# 🔐🤖 Smart Contract Vulnerability Detection — LLM Benchmark

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLMs-black.svg)
![Pandas](https://img.shields.io/badge/pandas-2.x-blue.svg)
![Scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)
![tqdm](https://img.shields.io/badge/tqdm-progress%20bar-yellow.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

<p align="center">
  <img src="https://media3.giphy.com/media/LlKN0pAfAduGRYWdbQ/giphy.gif" alt="Cyber Security Blockchain" width="600">
</p>

---

## 📝 Project Description

This repository is the **LLM zero-shot baseline** component of our research paper: **"Graph-Aware Transformers for Smart Contract Vulnerability Detection"** (ECE Paris, April 2026). Using **Ollama** to run local LLMs (llama3.2, deepseek-coder:6.7b, qwen2.5-coder:7b, TinyLlama, etc.), we benchmark how well off-the-shelf generative models can classify **Solidity smart contracts** into vulnerability categories — without any fine-tuning. The results form the baseline that our **GraphCodeBERT** fine-tuned model (see [graphs-ppe](https://github.com/axelbrons/graphs-ppe)) significantly outperforms, proving the need for structure-aware, embedding-based approaches in blockchain security.

I tried to improve the best LLM in [**this LoRA code**](https://github.com/Thibault-GAREL/PPE_LoRa_Smart_contract).

Check the link to see the paper and all the docs !

<details>
<summary>⚙️ <b>Feature list</b></summary>

- **Zero-shot prompting** pipeline, sends raw Solidity code to any Ollama-compatible LLM and extracts a single digit prediction
- **Flexible classification**: supports **4-label** (Delegatecall, Overflow, Reentrancy, Timestamp) and **8-label** (+ Block Number, Ether Frozen, Strict Equality, Unchecked Call) dataset configs
- **Automatic evaluation**: computes precision, confusion matrix, and per-class breakdown after each run
- **CSV export**: saves all predictions alongside ground-truth labels for offline analysis
- **Multi-model support**: swap between `llama3.2`, `deepseek-coder:6.7b`, `qwen2.5-coder:7b`, `tinyllama:1.1b`, `codellama`, `mistral` via a single config line
- **Progress tracking**: `tqdm` progress bar with per-contract timing
- **Research-grade results**: data used directly in our published paper comparing LLMs vs. fine-tuned GraphCodeBERT

</details>

---

## Example Outputs

### 📊 LLM Benchmark — Paper Results (Table 2)

> Extracted from our research paper. Models tested on a subset of the 8-label dataset via zero-shot prompting. *Italic* = fine-tuned with LoRA.

| Model                   | Time (min:sec) | Precision (%) |
|-------------------------|----------------|---------------|
| llama3.2                | 22:48          | **34.00**     |
| llama3.2                | 17:33          | 29.47         |
| llama3.2                | 09:03          | 24.00         |
| *TinyLlama-1.1B + LoRA* | —              | *22.20*       |
| deepseek-coder:6.7b     | 43:57          | 6.25          |
| kimi-k2.5:cloud         | 47:14          | 6.00          |
| qwen2.5-coder:7b        | —              | 4.00          |
| codellama               | 32:19          | 2.13          |
| mistral                 | 26:16          | 2.00          |

📄 Full paper with GraphCodeBERT results (87% F1, 91% recall): [graphs-ppe](https://github.com/axelbrons/graphs-ppe) — also available locally as [`ppe-paper-last-version.pdf`](ppe-paper-last-version.pdf)

### 📝 Notes & Observations

- Best precision achieved: **34%** (llama3.2) — far below practical security standards
- Code-specialized models (deepseek-coder, qwen2.5-coder) surprisingly performed **worse** than general-purpose LLMs on this task
- Inference time: 9 to 47+ minutes for just ~50–100 contracts → impractical for real-world auditing
- Root cause: LLMs suffer from **"Protected Mechanism Bias"** — they recognize security keywords (`onlyOwner`, `require`) without understanding the underlying execution logic
- These results motivated our full paper and the move to fine-tuned **GraphCodeBERT** with Data Flow Graph integration → 87% F1-score, 91% recall

---

## 🗺️ How it works

```text
┌─────────────────────┐
│  CSV Dataset        │  SC_4label.csv / SC_Vuln_8label.csv
│  (Solidity code     │
│  + ground-truth)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Zero-Shot Prompt   │  "Analyze this Solidity contract.
│  Builder            │   Reply ONLY with a digit 0-8."
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Ollama Local API   │  llama3.2 / deepseek-coder:6.7b
│  localhost:11434    │  qwen2.5-coder:7b / tinyllama ...
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Response Parser    │  Extract first valid digit
│                     │  Invalid → -1 (error)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Evaluation         │  Precision, Confusion Matrix,
│  + CSV Export       │  Classification Report
└─────────────────────┘
```

**Vulnerability labels (8-class setup):**

| ID | Label | Description               |
|----|-------|---------------------------|
|  0 | BN    | Block Number Dependency   |
|  1 | DE    | Dangerous Delegatecall    |
|  2 | EF    | Ether Frozen              |
|  3 | SE    | Ether Strict Equality     |
|  4 | OF    | Integer Overflow          |
|  5 | RE    | Reentrancy                |
|  6 | TP    | Timestamp Dependency      |
|  7 | UC    | Unchecked External Call   |
|  8 | —     | Normal (no vulnerability) |

- The CSV dataset (`SC_4label.csv` or `SC_Vuln_8label.csv`) is loaded via `csv_loader.py`, which handles Solidity code containing commas and special characters
- For each contract, `llm_classifier_v2.py` builds a **zero-shot prompt** asking the model to output a single digit (0–4 or 0–8) matching the vulnerability type
- The prompt is sent to the **Ollama local API** (`http://localhost:11434`) via an HTTP POST request, no internet required
- The raw LLM response is parsed to extract the first valid digit; invalid responses are flagged as `-1` (error)
- After all contracts are classified, precision, per-class breakdown, and a full confusion matrix are computed and printed
- Final results (ground-truth label + LLM prediction) are exported to a CSV file for further analysis

---

## 📂 Repository structure

```bash
├── archive/
│   ├── SC_4label.csv              # Dataset with 4 vulnerability types
│   └── SC_Vuln_8label.csv         # Dataset with 8 vulnerability types
│
├── img/                           # Result screenshots for README
│   ├── img.png ... img_8.png
│
├── csv_loader.py                  # Robust CSV loader for Solidity code
├── llm_classifier_v2.py           # Prompt builder + Ollama API client
│
├── main_v2.py                     # ✅ LATEST VERSION — flexible config (4 or 8 labels, any model)
├── main_v1.py                     # ⚠️  Legacy — 4 labels only, hardcoded params, no confusion matrix
│
├── resultats_classification.csv          # Output: 4-label run
├── resultats_classification_8labels.csv  # Output: 8-label run
│
├── ppe-paper-last-version.pdf     # 📄 Full research paper (ECE Paris, April 2026)
├── LICENSE
└── README.md
```

---

## 💻 Run it on Your PC

> ⚠️ You need **[Ollama](https://ollama.com/)** installed and running locally. Pull a model first: `ollama pull llama3.2`

Clone the repository and install dependencies:

```bash
git clone https://github.com/Thibault-GAREL/PPE-Smart_contract-LLM_Test.git
cd PPE-Smart_contract-LLM_Test

python -m venv .venv  # if you don't have a virtual environment
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install pandas tqdm scikit-learn requests
```

### Configure and run

Edit the `CONFIG` block at the top of `main_v2.py`:

```python
CONFIG = {
    "dataset": "8labels",       # "4labels" or "8labels"
    "limite": 100,              # number of contracts to classify (None = all)
    "model": "llama3.2",        # any model pulled via Ollama
}
```

Then run:

```bash
python main_v2.py
```

### Available models (tested)

```bash
ollama pull llama3.2
ollama pull deepseek-coder:6.7b
ollama pull qwen2.5-coder:7b
ollama pull tinyllama:1.1b
ollama pull codellama
ollama pull mistral
```

---

## 📖 Inspiration / Sources

This project is part of our **PPE research project** at ECE Paris (Majeure Data & IA, 2026). The LLM benchmark in this repo feeds directly into our paper:

- 📄 [Graph-Aware Transformers for Smart Contract Vulnerability Detection](https://github.com/axelbrons/graphs-ppe) — full paper + GraphCodeBERT fine-tuning code
- 📦 [Kaggle dataset by TranDuongMinhDai](https://www.kaggle.com/) — source of the 8-label vulnerability dataset
- 🔬 [Chen et al., 2025 — When ChatGPT meets smart contract vulnerability detection](https://dl.acm.org/doi/10.1145/3695988) — key reference for LLM limitations in security

[Reasearch paper](https://github.com/Thibault-GAREL/PPE_LLM_test_Smart_contract/blob/main/ppe-paper-last-version.pdf) mainly done by  [Axel Bröns](https://github.com/axelbrons) and [Me](https://github.com/Thibault-GAREL)

I tried to improve the best LLM with a LoRA in [this code](https://github.com/Thibault-GAREL/PPE_LoRa_Smart_contract)

Code created by me 😎, Thibault GAREL — [Github](https://github.com/Thibault-GAREL)
