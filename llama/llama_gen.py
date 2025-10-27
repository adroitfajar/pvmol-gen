import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from rdkit import Chem
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = "./trained_llama_model"  # Must match training output
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float32  # Use FP32 for generation
)
model.eval()

def validate_smiles(smiles):
    """Validate SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            Chem.SanitizeMol(mol)
            return smiles
    except Exception as e:
        logger.info(f"Invalid SMILES: {smiles}, error: {e}")
    return None

def generate_smiles(target_num_samples):
    generated = set()
    attempts = 0
    device = next(model.parameters()).device
    # Use the BOS token if available; otherwise, fallback to the PAD token.
    start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
    while len(generated) < target_num_samples and attempts < target_num_samples * 100:
        input_ids = torch.tensor([[start_token]]).to(device)
        attention_mask = torch.ones(input_ids.shape, device=device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=100,
                num_return_sequences=1,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
        smiles = tokenizer.decode(outputs[0], skip_special_tokens=True)
        valid = validate_smiles(smiles)
        if valid and valid not in generated:
            generated.add(valid)
            logger.info(f"Generated: {valid}")
        attempts += 1
    return list(generated)

generated_smiles = generate_smiles(10000)
df = pd.DataFrame(generated_smiles, columns=["smiles"])
df.to_csv("generated_llama2.csv", index=False)
print("Generation complete!")
