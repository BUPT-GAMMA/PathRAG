import os
import asyncio
import torch
from PathRAG.RAGRunner import RAGRunner

if __name__ == "__main__":
    backend = "ms"  # hf / vllm / ollama / ms / local
    working_dir = f"./dickens_{backend}"
    llm_model_name = "Qwen/Qwen3-0.6B"  # 聊天模型名称或路径
    embedding_model_name = "iic/nlp_corom_sentence-embedding_english-base"  # 编码模型名称或路径

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
        embedding_dim=768,#编码器维度
        embedding_max_token_size=5000,
    )

    data_file = "各章小结.txt"
    question = "本文关于机器学习讲了什么内容，有何启发"

    with open(data_file, "r", encoding="utf-8") as f:
        runner.insert_text(f.read())

    answer = runner.query(question, mode="hybrid")
    print("问:", question)
    print("答:", answer)
