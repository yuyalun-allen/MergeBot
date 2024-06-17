import os
import time
import json
from functools import partial

import torch
from tqdm import tqdm
import fire
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
)
from peft import PeftModel
from codebleu import calc_codebleu
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)
from datasets import Features, Sequence, Value, load_dataset

from generate_dataset import load_merge_conflict_and_resolution_chunk


def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["conflict"], documents["resolution"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to("cuda"), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def main(**kwargs):
    # ctx_encoder = DPRContextEncoder.from_pretrained(rag_example_args.dpr_ctx_encoder_model_name).to("cuda")
    # ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(rag_example_args.dpr_ctx_encoder_model_name)
    # new_features = Features(
    #     {"conflict": Value("string"), "resolution": Value("string"), "embeddings": Sequence(Value("float32"))}
    # )  # optional, save as float32 instead of float64 to save space
    # dataset = dataset.map(
    #     partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
    #     batched=True,
    #     batch_size=processing_args.batch_size,
    #     features=new_features,
    # )

    # # And finally save your dataset
    # passages_path = os.path.join(rag_example_args.output_dir, "my_knowledge_dataset")
    # dataset.save_to_disk(passages_path)
    # # from datasets import load_from_disk
    # # dataset = load_from_disk(passages_path)  # to reload the dataset

    # ######################################
    # logger.info("Step 2 - Index the dataset")
    # ######################################

    # # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    # index = faiss.IndexHNSWFlat(index_hnsw_args.d, index_hnsw_args.m, faiss.METRIC_INNER_PRODUCT)
    # dataset.add_faiss_index("embeddings", custom_index=index)

    # # And save the index
    # index_path = os.path.join(rag_example_args.output_dir, "my_knowledge_dataset_hnsw_index.faiss")
    # dataset.get_index("embeddings").save(index_path)
    # # dataset.load_faiss_index("embeddings", index_path)  # to reload the index

    # ######################################
    # logger.info("Step 3 - Load RAG")
    # ######################################

    # # Easy way to load the model
    # retriever = RagRetriever.from_pretrained(
    #     rag_example_args.rag_model_name, index_name="custom", indexed_dataset=dataset
    # )
    # model = RagSequenceForGeneration.from_pretrained(rag_example_args.rag_model_name, retriever=retriever)
    # tokenizer = RagTokenizer.from_pretrained(rag_example_args.rag_model_name)
    torch.cuda.manual_seed(42)
    model = LlamaForCausalLM.from_pretrained(
        kwargs["model_name"],
        load_in_8bit=True,
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

    with open("filtered_chunks.json", "r") as f:
        c_n_r_chunks  = json.load(f)
    c_n_r_chunks = c_n_r_chunks[int(len(c_n_r_chunks)*0.999) :]
    # c_n_r_chunks = c_n_r_chunks[:int(len(c_n_r_chunks)*0.001)]
    evaluation(model=model, eval_data=c_n_r_chunks, tokenizer=tokenizer)

def evaluation(model, eval_data, tokenizer):
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
        prompt = (f"""
Output the the resolution code of this merge conflict without any explanation:

```
{batch['conflict']}
```

Resolution:

```
        """)
        prompt_ids = tokenizer(prompt, return_tensors="pt")
        prompt_ids = {k: v.to("cuda") for k, v in prompt_ids.items()}
        codebleu_score = []

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **prompt_ids,
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
        output_text = tokenizer.decode(outputs[0][torch.sum(prompt_ids['attention_mask']).item()+1:], skip_special_tokens=True)
        codebleu_result = calc_codebleu([batch['resolution']], [output_text], lang="java", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=tokenizer)
        codebleu_score.append(codebleu_result['codebleu'])
        with open(f"results/{id}.txt", "w") as f:
            f.write(f"######Resolution output:\n{output_text}\n######Reference resolution\n{batch['resolution']}\n######codebleu result:\n{codebleu_result}")
        print(codebleu_result['codebleu'])
        
    with open(f"results/average_codebleu.txt", "w") as f:
        f.write(f"codebleu result:\n{sum(codebleu_score) / len(codebleu_score)}")



if __name__ == "__main__":
    fire.Fire(main)