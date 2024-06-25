import time
import json
from functools import partial

import torch
from tqdm import tqdm
import fire
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from codebleu import calc_codebleu
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import Levenshtein


def embed(chunk: dict, model: SentenceTransformer) -> dict:
    """Compute the SentenceTransformer embeddings of document passages"""
    # 拼接标题和内容为一个输入
    texts = [f"{chunk['conflict']}"]
    # 生成嵌入
    embeddings = model.encode(texts, convert_to_tensor=True)
    # 返回原始字段和新生成的嵌入
    return {
        "conflict": chunk["conflict"],
        "resolution": chunk["resolution"],
        "embeddings": embeddings.cpu().numpy()
    }


def create_index(ST: SentenceTransformer):
    with open("filtered_chunks.json", "r") as f:
        c_n_r_chunks  = json.load(f)
    c_n_r_chunks_index = c_n_r_chunks[: int(len(c_n_r_chunks)*0.8)]
    chunks_dataset = Dataset.from_list(c_n_r_chunks_index)
    chunks_dataset.map(
        partial(embed, model=ST),
        batched=True,
        batch_size=16
    )
    chunks_dataset.add_faiss_index("embeddings")
    chunks_dataset.to_json("embedding_chunks.json")
    return chunks_dataset


def search(query: str, model: SentenceTransformer, dataset: Dataset, k: int = 3 ):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = model.encode(query) # embed new query
    scores, retrieved_examples = dataset.get_nearest_examples( # retrieve results
        "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
        k=k # get only top k results
    )
    return scores, retrieved_examples


def main(**kwargs):
    with open("filtered_chunks.json", "r") as f:
        c_n_r_chunks  = json.load(f)
    c_n_r_chunks_eval = c_n_r_chunks[int(len(c_n_r_chunks)*0.999) :]

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
    ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    index_dateset = create_index(ST)

    # c_n_r_chunks = c_n_r_chunks[:int(len(c_n_r_chunks)*0.001)]
    evaluation(model=model, eval_data=c_n_r_chunks_eval, tokenizer=tokenizer, index=index_dateset, ST=ST)


def get_evaluation_metrics(ref, pred):
    """
    Calculate various scores between reference and prediction
    Args:
        ref: reference resolution
        pred: predicted resolution
    Returns:
        various scores between reference and prediction
    """
    codebleu = calc_codebleu(ref, pred, lang="java")['codebleu']
    bleu = sentence_bleu([ref.split()], pred.split())
    my_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = my_rouge_scorer.score(ref, pred)['rougeL'].fmeasure
    levenshtein = Levenshtein.distance(ref, pred)

    return {
        "ref": ref,
        "pred": pred,
        "codebleu": codebleu,
        "bleu": bleu,
        "rouge": rouge,
        "levenshtein": levenshtein
    }


def evaluation(model, eval_data, tokenizer, index, ST):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        tokenizer: The tokenizer used to decode predictions

    Returns: Accuracy, CodeBLEU
    """
    results = []
    for _, batch in enumerate(tqdm(eval_data,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
        SYS_PROMPT = """You are an assistant for resolving merge conflicts with given conflict chunks.
        You are first given several resolution examples following with an unresolved conflict chunk. Provide the resolution of the conflict chunk.
        Only output the resolution and do not output any explanation."""
        prompt = ""
        _ , retrieved_chunks = search(batch['conflict'], ST, index,  5)
        for i, chunk in enumerate(retrieved_chunks):
            prompt += (f"""
Resolved Conflict {i}:

```
{chunk['conflict']}
```

Resolution:

```
{chunk['resolution']}
```

""")
        prompt = (f"""
Unresolved Conflict:

```
{batch['conflict']}
```

Resolution:

        """)
        messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":prompt}]
        # tell the model to generate
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
            ).to(model.device)

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
        result = get_evaluation_metrics(batch['resolution'], resolution)
        results.append(result)
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)



if __name__ == "__main__":
    fire.Fire(main)