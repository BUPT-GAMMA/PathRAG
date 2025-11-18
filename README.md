The code for the paper **"PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths"**.
## Install
```bash
cd PathRAG
pip install -e . # or pip install -r requirements.txt 
```

## RUN the project

### Windows

```bash
python -m venv .venv    # create virtual environment
.venv\Scripts\activate  # activate the virtual environment
python v1_test.py       # to run the project

# if it doesn't works properly then try reinstalling the packages using the above installation command
```

### Linux/Unix

```bash
python3 -m venv .venv    # create virtual environment
Source .venv\bin\activate  # activate the virtual environment
python3 v1_test.py       # to run the project

# if it doesn't works properly then try reinstalling the packages using the above installation command
```

## Quick Start
* You can quickly experience this project in the `v1_test.py` file.
* Rename `exampe.env` to `.env`
* Set OpenAI API key in `.env` file and the BASE URL.
* Prepare your retrieval document `text.txt`. You can modify this in the code in `v1_test.py`.
* The `v1_text.py` file is the entry point to initialize PathRAG and perform queries.
  
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
