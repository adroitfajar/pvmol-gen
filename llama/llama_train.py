import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from rdkit import Chem

print("Available GPUs:", torch.cuda.device_count())

# ----- Step 1: Load and Augment SMILES Data -----
data = pd.read_csv("class1_oa0.csv")  # Replace with your dataset path
smiles_list = data["smiles"].tolist()

def augment_smiles(smiles, num_augmentations=5):
    """Generate different representations of a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [smiles]
    augmented = set()
    augmented.add(Chem.MolToSmiles(mol, canonical=True))
    for _ in range(num_augmentations):
        augmented.add(Chem.MolToSmiles(mol, canonical=False))
    return list(augmented)

augmented_smiles_list = []
for s in smiles_list:
    augmented_smiles_list.extend(augment_smiles(s))

# ----- Step 2: Load the Tokenizer -----
model_name = "meta-llama/Llama-2-7b-hf"  # or use "meta-llama/Llama-2-13b-hf" if desired
token = "your_token"  # Your Hugging Face token
tokenizer = LlamaTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "right"

# Tokenize the SMILES strings
encoded_inputs = tokenizer(
    augmented_smiles_list,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# ----- Step 3: Create a Custom Dataset -----
class SmilesDataset(Dataset):
    def __init__(self, encoded_inputs):
        self.input_ids = encoded_inputs["input_ids"]
        self.attention_mask = encoded_inputs["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }

dataset = SmilesDataset(encoded_inputs)
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

# ----- Step 4: Load the Model in Full Precision (FP32) -----
# Since memory is not a worry on your supercomputer, we use FP32.
max_memory = {i: "32GB" for i in range(torch.cuda.device_count())}
model = LlamaForCausalLM.from_pretrained(
    model_name,
    token=token,
    device_map="auto",
    max_memory=max_memory,
    torch_dtype=torch.float32  # FP32 training
)
model.resize_token_embeddings(len(tokenizer))

# ----- Step 5: Define Training Arguments -----
training_args = TrainingArguments(
    output_dir="./llama_results",
    num_train_epochs=6,                     # Adjust epochs as needed
    per_device_train_batch_size=1,          # Use a small batch size for stability
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,          # Simulate a larger effective batch size
    fp16=False,                             # Full precision training
    max_grad_norm=1.0,                      # Enable gradient clipping
    warmup_steps=500,
    learning_rate=1e-5,                     # Low learning rate for stability
    weight_decay=0.01,
    logging_dir="./llama_logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=3,
)

early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[early_stopping],
)

# ----- Step 6: Train and Save the Model -----
trainer.train()

model.save_pretrained("./trained_llama_model")
tokenizer.save_pretrained("./trained_llama_model")

print("LLaMA-2 model training complete! Model and tokenizer saved.")
