import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete, ms_model_complete,ms_embedding
from PathRAG.utils import EmbeddingFunc
from PathRAG.merge import merge
import modelscope as ms
import torch
import asyncio

def main():
    WORKING_DIR = f"./dickens_merge"
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    tokenizer = ms.AutoTokenizer.from_pretrained(
        "iic/nlp_corom_sentence-embedding_english-base",
        trust_remote_code=True
    )
    model = ms.AutoModel.from_pretrained(
        "iic/nlp_corom_sentence-embedding_english-base",
        trust_remote_code=True
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    rag = PathRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ms_model_complete,
        llm_model_name="Qwen/Qwen3-0.6B",
        embedding_func=EmbeddingFunc(
            embedding_dim=768,  
            max_token_size=5000,
            func=lambda texts: ms_embedding(
                texts,
                tokenizer,
                model, 
            ),
        ),
    )
    # rag = PathRAG(
    #     working_dir=WORKING_DIR,
    #     llm_model_func=gpt_4o_mini_complete,
    # )
    data_file = "测试.txt"
    with open(data_file, "r", encoding="utf-8") as f:
        rag.insert(f.read()) 
    asyncio.run(merge(file_path=f"{WORKING_DIR}", rag=rag, stname='all-mpnet-base-v2'))

if __name__ == "__main__":
    main()
