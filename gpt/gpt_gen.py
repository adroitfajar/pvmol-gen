import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from rdkit import Chem
from rdkit.Chem import Draw
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./trained_model')
tokenizer = GPT2Tokenizer.from_pretrained('./trained_model')

# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def validate_smiles(smiles):
    """ Validate SMILES string using RDKit """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            Chem.SanitizeMol(mol)
            return smiles
    except Exception as e:
        logger.debug(f"Invalid SMILES: {smiles}, Error: {e}")
    return None

def generate_smiles(target_num_samples):
    generated_smiles = set()
    attempts = 0
    while len(generated_smiles) < target_num_samples and attempts < target_num_samples * 100:
        input_ids = torch.tensor(tokenizer.encode('[PAD]', add_special_tokens=False)).unsqueeze(0).to(device)
        attention_mask = torch.ones(input_ids.shape, device=device)  # Ensure attention mask is set
        try:
            sample_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=100,
                num_return_sequences=1,
                temperature=0.9,  # Increase temperature for more diversity
                do_sample=True
            )
            smiles = tokenizer.decode(sample_output[0], skip_special_tokens=True)
            valid_smiles = validate_smiles(smiles)
            if valid_smiles and valid_smiles not in generated_smiles:
                generated_smiles.add(valid_smiles)
                logger.info(f"Generated valid SMILES: {valid_smiles}")
            else:
                logger.debug(f"Invalid or duplicate SMILES: {smiles}")
        except Exception as e:
            logger.error(f"Error during SMILES generation: {e}")
        attempts += 1
    if len(generated_smiles) < target_num_samples:
        logger.warning(f"Only {len(generated_smiles)} valid SMILES were generated after {attempts} attempts.")
    return list(generated_smiles)

# Generate 100000 new and valid unique SMILES strings
new_smiles = generate_smiles(100000)
logger.info(f"Generated {len(new_smiles)} unique valid SMILES")

# Save generated SMILES to a CSV file
df = pd.DataFrame(new_smiles, columns=['smiles'])
df.to_csv('generated0.csv', index=False)
logger.info("Generated SMILES saved to generated0.csv")

# # Visualize each molecule and save as individual images
# if not os.path.exists('molecule_images'):
#     os.makedirs('molecule_images')

# image_paths = []
# for i, smiles in enumerate(new_smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol:
#         img = Draw.MolToImage(mol, size=(200, 200))
#         img_path = f'molecule_images/molecule_{i+1}.png'
#         img.save(img_path)
#         image_paths.append(img_path)

print("Generation completed!")
