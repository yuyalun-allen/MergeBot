import tqdm
import torch
import fire
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM
)

from generate_dataset import preprocess_merge_conflict_and_resolution

def main(**kwargs):
    model = LlamaForCausalLM.from_pretrained(
        kwargs["model_name"],
        load_in_8bit=True,
        device_map="auto",
    )

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(kwargs["model_name"] if kwargs["tokenizer_name"] is None else kwargs["tokenizer_name"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    dataset_val = preprocess_merge_conflict_and_resolution(dataset_config=None, tokenizer=tokenizer, split="test")
    eval_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        num_workers=1,
        pin_memory=True,
    )
    if len(eval_dataloader) == 0:
        raise ValueError("The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set.")
    else:
        print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")
    
    evaluation(model=model, eval_dataloader=eval_dataloader, tokenizer=tokenizer)

def evaluation(model, eval_dataloader, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        tokenizer: The tokenizer used to decode predictions

    Returns: Accuracy, CodeBLEU
    """
    model.eval()
    eval_preds = []
    total_eval_steps = 0
    for _, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
        total_eval_steps += 1
        for key in batch.keys():
            batch[key] = batch[key].to('cuda:0')
        # Ensure no gradients are computed for this scope to save memory
        with torch.no_grad():
            # Forward pass and compute loss
            outputs = model(**batch)

        # Decode predictions and add to evaluation predictions list
        preds = torch.argmax(outputs.logits, -1)
        eval_preds.extend(
            tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
        )


if __name__ == "__main__":
    fire.Fire(main)