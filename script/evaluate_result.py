import os
import time
import json

import torch
from tqdm import tqdm
import fire
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from codebleu import calc_codebleu
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from generate_dataset import load_merge_conflict_and_resolution_chunk


def main(**kwargs):
    with open("filtered_chunks.json", "r") as f:
        c_n_r_chunks  = json.load(f)
    c_n_r_chunks_index = c_n_r_chunks[: int(len(c_n_r_chunks)*0.8)]
    c_n_r_chunks_eval = c_n_r_chunks[int(len(c_n_r_chunks)*0.999) :]

    ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    chunks_embeddings = ST.encode(c_n_r_chunks_index)
    chunks_embeddings = np.array(chunks_embeddings).astype('float32')
    index = faiss.IndexFlatL2(chunks_embeddings.shape[1])
    index.add(chunks_embeddings)

    # use quantization to lower GPU usage
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    torch.cuda.manual_seed(42)
    model = LlamaForCausalLM.from_pretrained(
        kwargs["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        return_dict=True,
    )

    if kwargs["from_peft_checkpoint"]:
        model = PeftModel.from_pretrained(model, kwargs["from_peft_checkpoint"], is_trainable=True)

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(kwargs["model_name"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    # c_n_r_chunks = c_n_r_chunks[:int(len(c_n_r_chunks)*0.001)]
    evaluation(model=model, eval_data=c_n_r_chunks_eval, tokenizer=tokenizer)

def retrieve_docs(query, retriever_model, index, corpus_texts, top_k=5):
    query_embedding = retriever_model.encode([query]).astype('float32')
    _, indices = index.search(query_embedding, top_k)
    return [corpus_texts[idx] for idx in indices[0]]


def evaluation(model, eval_data, tokenizer, retriever_model, index, corpus_text):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        tokenizer: The tokenizer used to decode predictions

    Returns: Accuracy, CodeBLEU
    """
    model.eval()
    os.makedirs("results", exist_ok=True)
    for id, batch in enumerate(tqdm(eval_data,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
        SYS_PROMPT = """You are an assistant for resolving merge conflicts with given conflict chunks.
        You are first given an unresolved conflict chunk following with several resolution examples. Provide the resolution of the conflict chunk.
        Only output the resolution and do not output any explanation."""
        prompt = (f"""
Conflict:

```
{batch['conflict']}
```

Resolution:

        """)
        _ , retrieved_documents = retrieve_docs(prompt, retriever_model, index, corpus_text, 5)
        messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":prompt}]
        # tell the model to generate
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
            ).to(model.device)
        codebleu_score = []

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=True,
                top_p=1.0,
                temperature=1.0,
                use_cache=True,
                top_k=50,
                repetition_penalty=1.0,
                length_penalty=1
            )
        e2e_inference_time = (time.perf_counter()-start)*1000
        print(f"the inference time is {e2e_inference_time} ms")
        # Only get the output text without the prompt text
        # output_text = tokenizer.decode(outputs[0][torch.sum(input_ids['attention_mask']).item()+1:], skip_special_tokens=True)
        output_text: str = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        resolution = output_text.split('\n')
        if len(resolution) > 2:
            resolution = '\n'.join(resolution[1:-1])
        else:
            resolution = ''
        codebleu_result = calc_codebleu([batch['resolution']], [resolution], lang="java", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=tokenizer)
        codebleu_score.append(codebleu_result['codebleu'])
        with open(f"results/{id}.txt", "w") as f:
            f.write(f"######Resolution output:\n{resolution}\n######Reference resolution\n{batch['resolution']}\n######codebleu result:\n{codebleu_result}")
        print(codebleu_result['codebleu'])
        
    with open(f"results/average_codebleu.txt", "w") as f:
        f.write(f"codebleu result:\n{sum(codebleu_score) / len(codebleu_score)}")



if __name__ == "__main__":
    fire.Fire(main)