import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import AgglomerativeClustering

# 1. Load the precomputed CSV
df = pd.read_csv('properties.csv')

# 2. Apply your filters (adjust numeric ranges as desired)
filtered = df[
    (df['SA'] <= 6) &
    (df['PAINS'] == False) &
    (df['HBA'].between(2, 5)) &
    (df['HBD'].between(0, 2)) &
    (df['TPSA'].between(50, 120)) &
    (df['Gap'].between(1.5, 5.0)) &
    (df['Dipole'].between(1.5, 4.0))
].copy()

print(f"After filtering: {len(filtered)} molecules")

# 3. If you want exactly 10 diverse picks, cluster Morgan fingerprints
if len(filtered) >= 10:
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024)
        for s in filtered['SMILES']
    ]
    arr = np.array([list(fp) for fp in fps])
    labels = AgglomerativeClustering(n_clusters=10).fit_predict(arr)
    filtered['Cluster'] = labels
    selected = filtered.groupby('Cluster').first().reset_index()
else:
    print("Fewer than 10 molecules passed filters; skipping clustering.")
    selected = filtered.copy()

# 4. Save the final picks
selected.to_csv('selected_molecules.csv', index=False)
print("Saved selected_molecules.csv")