import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback
from rdkit import Chem

# Step 1: Prepare the Data
data = pd.read_csv('class1_oa0.csv')
smiles_list = data['smiles'].tolist()

# Augment SMILES
def augment_smiles(smiles, num_augmentations=5):
    """ Generate different representations of a SMILES string """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [smiles]
    
    augmented_smiles = set()
    augmented_smiles.add(Chem.MolToSmiles(mol, canonical=True))
    
    for _ in range(num_augmentations):
        augmented_smiles.add(Chem.MolToSmiles(mol, canonical=False))
    
    return list(augmented_smiles)

# Apply augmentation
augmented_smiles_list = []
for smiles in smiles_list:
    augmented_smiles_list.extend(augment_smiles(smiles))

# Tokenization using GPT2 Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Encode the SMILES strings
encoded_inputs = tokenizer(augmented_smiles_list, padding=True, truncation=True, return_tensors="pt")

# Create a custom Dataset
class SmilesDataset(Dataset):
    def __init__(self, encoded_inputs):
        self.input_ids = encoded_inputs['input_ids']
        self.attention_mask = encoded_inputs['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx]
        }

dataset = SmilesDataset(encoded_inputs)

# Split the data
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Step 2: Build and Train the Model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,  # Max number of epochs
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",  # Evaluate every epoch
    save_strategy="epoch",
    load_best_model_at_end=True,  # Load the best model at the end of training
)

# Adding EarlyStoppingCallback
early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    callbacks=[early_stopping],  # Add early stopping callback
)

trainer.train()

# Save the trained model and tokenizer
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
print("Model and tokenizer saved.")
