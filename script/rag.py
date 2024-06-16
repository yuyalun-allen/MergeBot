import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from threading import Thread
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

dataset = load_dataset("not-lain/wikipedia",revision = "embedded")

data = dataset["train"]
data = data.add_faiss_index("embeddings") # column name that has the embeddings of the dataset


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# use quantization to lower GPU usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

SYS_PROMPT = """You are an assistant for resolving merge conflicts with given conflict chunks.
You are first given an unresolved conflict chunk following with several resolution examples. Provide the resolution of the conflict chunk.
Be concise, only output the resolution and do not output any explanation."""


def search(query: str, k: int = 3 ):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = ST.encode(query) # embed new query
    scores, retrieved_examples = data.get_nearest_examples( # retrieve results
        "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
        k=k # get only top k results
    )
    return scores, retrieved_examples


def format_prompt(prompt,retrieved_documents,k):
    """using the retrieved documents we will prompt the model to generate our responses"""
    PROMPT = f"Conflicts:\n{prompt}\nExamples:\n"
    for idx in range(k) :
        PROMPT+= f"{retrieved_documents['text'][idx]}\n"
    return PROMPT


@spaces.GPU(duration=150)
def talk(prompt):
    k = 1 # number of retrieved documents
    _ , retrieved_documents = search(prompt, k)
    formatted_prompt = format_prompt(prompt,retrieved_documents,k)
    formatted_prompt = formatted_prompt[:2000] # to avoid GPU OOM
    messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]
    # tell the model to generate
    input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
      input_ids,
      max_new_tokens=256,
      eos_token_id=terminators,
      do_sample=True,
      top_k=50,
      temperature=1.0,
      top_p=1.0,
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
