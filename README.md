The code for the paper **"PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths"**.
## Install
```bash
cd PathRAG
pip install -e .
```
## Quick Start
* You can quickly experience this project in the `v1_test.py` file.
* Set OpenAI API key in environment if using OpenAI models: `api_key="sk-...".` in the `v1_test.py` and `llm.py` file
* Prepare your retrieval document "text.txt".
* Use the following Python snippet in the "v1_text.py" file to initialize PathRAG and perform queries.
  
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
## Parameter modification
You can adjust the relevant parameters in the `base.py` and `operate.py` files.

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

## Graph Visualization
Once your files were processed, you may visualize how the graph looks like.
1. Ensure the file `graph_chunk_entity_relation.graphml` has been written during the main indexation. 
2. Run `visualize_graph.py`. 

It will create an HTML of your graph. Feel free to open it on your favorite browser.
* The edge labels are the relationship strength.
* All nodes have tooltips with their category, description and chunk source.


## Cite
Please cite our paper if you use this code in your own work:
```python
@article{chen2025pathrag,
  title={PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths},
  author={Chen, Boyu and Guo, Zirui and Yang, Zidan and Chen, Yuluo and Chen, Junze and Liu, Zhenghao and Shi, Chuan and Yang, Cheng},
  journal={arXiv preprint arXiv:2502.14902},
  year={2025}
}
```
