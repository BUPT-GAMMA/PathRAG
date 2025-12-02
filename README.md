<div align ="center">
  <h2>
  PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths
  </h2>
</div>
<div align="center">
  <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 25px; text-align: center;">
    <p>
      <a href='https://github.com/BUPT-GAMMA/PathRAG'><img src='https://img.shields.io/badge/🔥Project-Page-00d9ff?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e'></a>
      <a href='https://arxiv.org/abs/2502.14902'><img src='https://img.shields.io/badge/📄arXiv-2410.05779-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e'></a>
    </p>
  </div>
</div>
<img width="2654" height="1348" alt="image" src="https://github.com/user-attachments/assets/c75aa831-5f43-42b2-8871-149e91b139e1" />

## Install
Python Version: Python 3.10.18
```bash
cd PathRAG
pip install -e .
```
## Quick Start
- use OpenAI API key 
* You can quickly experience this project in the `v1_test.py` file.
* Set OpenAI API key in environment if using OpenAI models: `api_key="sk-...".` in the `v1_test.py` and `llm.py` file
* Prepare your retrieval document "text.txt".
* Use the following Python snippet in the `v1_test.py` file to initialize PathRAG and perform queries.
  
```python
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete

WORKING_DIR = "./your_working_dir"
api_key="your_api_key"
os.environ["OPENAI_API_KEY"] = api_key
base_url="https://api.openai.com/v1"
os.environ["OPENAI_API_BASE"]=base_url


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  
)

data_file="./text.txt"
question="your_question"
with open(data_file) as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))
```
## Quick Start with models from different sources
- You can use the model from huggingface, ollama, modelscope, local and vllm
- You can quickly experience this project in the `rag_test.py` file
- Select your model source—— hf / vllm / ollama / ms / local
- Prepare your llm_model，embedding_model and  retrieval document "your data file" . 
- Use the following Python snippet in the `rag_test.py` file to use models from different sources
- Detailed examples can be referred to`不同模型样例.txt`
```Python
import os
import asyncio
import torch
from PathRAG.RAGRunner import RAGRunner
if __name__ == "__main__":
	backend = "your_model_source" # hf / vllm / ollama / ms / local可选
	working_dir = f"your_working_dir" # 工作目录
	llm_model_name = "Qwen/Qwen3-0.6B" # 聊天模型名称或路径
	embedding_model_name = "iic/nlp_corom_sentence-embedding_english-base" # 编码模型名称或路径
	# ollama 额外参数
	llm_model_kwargs = {
	"host": "http://localhost:11434",
	"options": {"num_ctx": 8192},
	"timeout": 300,
	} if backend == "ollama" else {}
	
	runner = RAGRunner(
		backend=backend,
		working_dir=working_dir,
		llm_model_name=llm_model_name,
		embedding_model_name=embedding_model_name,
		llm_model_max_token_size=8192,
		llm_model_kwargs=llm_model_kwargs,
		embedding_dim=768,
		embedding_max_token_size=5000,
	)
	data_file = "your_data_file"
	question = "your_question"
	with open(data_file, "r", encoding="utf-8") as f:
		runner.insert_text(f.read())
	answer = runner.query(question, mode="hybrid")
	print("问:", question)
	print("答:", answer)
```

## Parameter modification
You can adjust the relevant parameters in the `base.py` and `operate.py` files.
- You can change the hyperparameter `top_k` in `base.py`, where `top_k` represents the number of nodes retrieved
- You can change the hyperparameters `alpha` and `threshold` in `operate.py`, where `alpha` represents the decay rate of information propagation along the edges, and `threshold` is the pruning threshold.
## Batch Insert
```python
import os
folder_path = "your_folder_path"  

txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
for file_name in txt_files:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        rag.insert(file.read())
```

## Evaluation

### Dataset
The dataset used in PathRAG can be downloaded from [TommyChien/UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain).
### Eval Metrics
<details>
<summary> Prompt </summary>
  
```python
You will evaluate two answers to the same question based on five criteria: **Comprehensiveness**, **Diversity**, **logicality**, **Coherence**, **Relevance**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **logicality**: How logically does the answer respond to all parts of the question?
- **Coherence**: How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?
- **Relevance**: How relevant is the answer to the question, staying focused and addressing the intended topic or issue?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why.

Here is the question:
{query}

Here are the two answers:
**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the five criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:
{{
  "Comprehensiveness": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "Diversity": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "logicality": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "Coherence": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "Relevance": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }}
}}
```

</details>

<img width="1834" height="1290" alt="image" src="https://github.com/user-attachments/assets/9a036bbb-200a-44ad-a307-356129a991ff" />

## Contribution
Boyu Chen, Zirui Guo, Zidan Yang, Junfei Bao  
## Citation
Please cite our paper if you use this code in your own work:
```python
@article{chen2025pathrag,
    title={PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths}, 
    author={Boyu Chen and Zirui Guo and Zidan Yang and Yuluo Chen and Junze Chen and Zhenghao Liu and Chuan Shi and Cheng Yang},
    year={2025},
    eprint={2502.14902},
    archivePrefix={arXiv},
    primaryClass={cs.CL}, 
}
```
