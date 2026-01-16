import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import sascorer
import numpy as np

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# 1. Load SMILES
df = pd.read_csv('genclass1.csv')  # your input SMILES file
smiles_list = df['smiles'].tolist()

# 2. Prepare PAINS filter
pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog(pains_params)

def compute_rdkit_props(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Chem.MolSanitizeException:
        return None

    return {
        'SMILES': smiles,
        'SA': sascorer.calculateScore(mol),
        'PAINS': pains_catalog.HasMatch(mol),
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
    }

# 3. RDKit pre-filter (optional: skip if you want to compute xTB for all)
rdkit_records = []
for smi in smiles_list:
    r = compute_rdkit_props(smi)
    if r is not None:
        rdkit_records.append(r)
print(f"RDKit parsing completed: {len(rdkit_records)} valid molecules")

# 4. xTB single-point (parallel)
def compute_xtb(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    result_embed = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if result_embed != 0 or mol.GetNumConformers() == 0:
        return smiles, np.nan, np.nan

    conf = mol.GetConformer(0)
    coords = conf.GetPositions()
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        f.write(f"{mol.GetNumAtoms()}\n\n".encode())
        for atom, pos in zip(atoms, coords):
            f.write(f"{atom} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n".encode())
        xyz_path = f.name

    proc = subprocess.run(
        ["xtb", xyz_path, "--opt", "none"],
        capture_output=True,
        text=True
    )
    os.remove(xyz_path)

    gap = dip = np.nan
    for line in proc.stdout.splitlines():
        clean = line.replace("|", "").strip()
        if "HOMO-LUMO/GAP" in clean or ("HOMO-LUMO" in clean and "GAP" in clean):
            try:
                # adjust index based on format, “... : 2.842 eV”
                gap = float(clean.split()[-2])
            except:
                pass
        if "Total Dipole" in clean and "Debye" in clean:
            try:
                dip = float(clean.split()[-2])
            except:
                pass

    return smiles, gap, dip

max_workers = 40
xtb_results = {}

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(compute_xtb, rec['SMILES']): rec['SMILES']
        for rec in rdkit_records
    }
    for future in as_completed(futures):
        smi, gap, dip = future.result()
        xtb_results[smi] = (gap, dip)

# 5. Combine RDKit + xTB properties
records = []
for rec in rdkit_records:
    gap, dip = xtb_results.get(rec['SMILES'], (np.nan, np.nan))
    records.append({
        'SMILES': rec['SMILES'],
        'SA': rec['SA'],
        'PAINS': rec['PAINS'],
        'HBD': rec['HBD'],
        'HBA': rec['HBA'],
        'TPSA': rec['TPSA'],
        'Gap': gap,
        'Dipole': dip
    })

prop_df = pd.DataFrame(records)
prop_df.to_csv('all_properties.csv', index=False)
print("Wrote all_properties.csv with RDKit + xTB columns")

# Note:
# In case xTB produces too many NaN results for the dipole, replace it with the Gasteiger method: add_dipole.ipynb