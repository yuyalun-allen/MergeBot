import os
import time

import torch
from tqdm import tqdm
import fire
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
)
from peft import PeftModel
from codebleu import calc_codebleu

from generate_dataset import load_merge_conflict_and_resolution_chunk

def main(**kwargs):
    model = LlamaForCausalLM.from_pretrained(
        kwargs["model_name"],
        load_in_8bit=True,
        device_map="auto",
    )

    if kwargs["from_peft_checkpoint"]:
        model = PeftModel.from_pretrained(model, kwargs["from_peft_checkpoint"], is_trainable=True)

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(kwargs["model_name"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    c_n_r_chunks = load_merge_conflict_and_resolution_chunk("output.json", "test")
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
        prompt = (
            f"Resolve this merge conflict:\n{batch['conflict']}\n---\nResolution:\n"
        )
        prompt_ids = tokenizer(prompt, padding='max_length', truncation=True, max_length=1024, return_tensors="pt")
        prompt_ids = {k: v.to("cuda") for k, v in prompt_ids.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **prompt_ids,
                max_new_tokens=1024,
                do_sample=True,
                top_p=1.0,
                temperature=1.0,
                use_cache=True,
                top_k=50,
                repetition_penalty=1.0,
                length_penalty=1,
            )
        e2e_inference_time = (time.perf_counter()-start)*1000
        print(f"the inference time is {e2e_inference_time} ms")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        codebleu_result = calc_codebleu(batch['resolution'], output_text, lang="java", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=tokenizer)
        print(f"codebleu result:\n{codebleu_result}")



if __name__ == "__main__":
    fire.Fire(main)