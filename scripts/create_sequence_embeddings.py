import pandas as pd
import numpy as np
import torch
import umap
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the metadata DataFrame in .csv format
df = pd.read_csv("data/processed/metadata.csv")

# Set up ProtT5
device   = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model     = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
model.to(device).eval()

# Function to batch embed sequences
def batch_embed(sequences: list[str], batch_size: int = 8) -> np.ndarray:
    all_embs = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch = sequences[i : i + batch_size]
        spaced = [" ".join(s) for s in batch]
        enc = tokenizer(spaced, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc)[0]      # (B, L, H)
        mask   = enc["input_ids"].ne(tokenizer.pad_token_id).unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)  # (B, H)
        all_embs.append(pooled.cpu().numpy())
    return np.vstack(all_embs)  # (N, H)

# Use the sequences from the DataFrame to create embeddings
sequences = df["sequence"].tolist()
embeddings = batch_embed(sequences, batch_size=8)  

# Save the embeddings
np.save("data/processed/04-with-sequence-embeddings/embeddings.npy", embeddings)

# embeddings is an (N, 1024) NumPy array, and df has N rows in the same order
df["sequence_embeddings"] = embeddings.tolist()

# Save the DataFrame with embeddings in pickle format
df.to_pickle("data/processed/04-with-sequence-embeddings/metadata_with_embeddings.pkl")