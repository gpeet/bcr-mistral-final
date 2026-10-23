from bcrm.paths import data_path, model_path, out_path, repo_path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MistralForCausalLM, PreTrainedTokenizerFast, MistralConfig
import pandas as pd
from tqdm import tqdm
import csv
from torch.cuda.amp import autocast

# === Dataset ===
class SequenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, sequence_col):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.sequence_col = sequence_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx][self.sequence_col]
        encoded = self.tokenizer.encode_plus(
            seq,
            max_length=512,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "index": idx,
        }


# === Weighted mean pooling ===
def weighted_mean_pooling(hidden_states, attention_mask):
    """
    Weighted mean pooling where weights increase with token position
    (1, 2, 3, ..., seq_len) and are masked for padding tokens.
    """
    batch_size, seq_len, _ = hidden_states.shape

    # Create position-based weights: shape [1, seq_len]
    position_weights = torch.arange(1, seq_len + 1, device=hidden_states.device).unsqueeze(0)

    # Apply attention mask
    weights = attention_mask * position_weights  # [batch, seq_len]

    # Weighted sum over tokens
    weighted_sum = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

    # Normalize by sum of weights (avoid divide-by-zero)
    denom = torch.sum(weights, dim=-1, keepdim=True).clamp(min=1e-6)

    return weighted_sum / denom  # [batch, hidden_size]


# === Inference with mean, max, and weighted mean pooling ===
def predict_and_extract_embeddings(
    model, tokenizer, input_csv,
    mean_output_csv, max_output_csv, weighted_output_csv,
    sequence_col, batch_size=32
):
    df = pd.read_csv(input_csv)
    dataset = SequenceDataset(df, tokenizer, sequence_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.cuda()

    # Prepare output headers
    header = ["index"] + [f"hidden_state_{i}" for i in range(model.config.hidden_size)]
    for path in [mean_output_csv, max_output_csv, weighted_output_csv]:
        with open(path, "w", newline="") as f_out:
            csv.writer(f_out).writerow(header)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            indices = batch["index"]

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
            mask = attention_mask.unsqueeze(-1)

            # === Mean pooling ===
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = (summed / counts).cpu().tolist()

            # === Max pooling ===
            masked_hidden = hidden_states.masked_fill(mask == 0, float('-inf'))
            max_pooled = masked_hidden.max(dim=1).values.cpu().tolist()

            # === Weighted mean pooling (position-weighted) ===
            weighted_pooled = weighted_mean_pooling(hidden_states, attention_mask).cpu().tolist()

            # Write outputs
            with open(mean_output_csv, "a", newline="") as f_mean, \
                 open(max_output_csv, "a", newline="") as f_max, \
                 open(weighted_output_csv, "a", newline="") as f_weighted:
                writer_mean = csv.writer(f_mean)
                writer_max = csv.writer(f_max)
                writer_weighted = csv.writer(f_weighted)

                for i, mean_row, max_row, weighted_row in zip(indices, mean_pooled, max_pooled, weighted_pooled):
                    writer_mean.writerow([i] + mean_row)
                    writer_max.writerow([i] + max_row)
                    writer_weighted.writerow([i] + weighted_row)


# === Load tokenizer and model ===
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=repo_path("foundation-models/base_only_bpe_tokenizer/tokenizer.json"),
    padding_side="right",
)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
vocab_size  = len(tokenizer)

configuration = MistralConfig(
    vocab_size=vocab_size, hidden_size=768, intermediate_size=3072,
    num_hidden_layers=16, num_attention_heads=12, num_key_value_heads=12,
    hidden_act="silu", max_position_embeddings=512, initializer_range=0.02,
    rms_norm_eps=1e-06, use_cache=True, pad_token_id=3,
    bos_token_id=None, eos_token_id=None, rope_theta=10000.0, sliding_window=512,
)
model  = MistralForCausalLM(configuration)
model.load_state_dict(
    torch.load(model_path("2025-HL-Foundation-300M-gradClip-undiversified.pth"))
)

# === Run embedding extraction ===
predict_and_extract_embeddings(
    model=model,
    tokenizer=tokenizer,
    input_csv=data_path("embedding-comparison/triplet_embedding_test_set_igblast.csv"),
    mean_output_csv=out_path("embeddings/mistral-meanpooled-embeddings-300M-undiversified.csv"),
    max_output_csv=out_path("embeddings/mistral-maxpooled-embeddings-300M-undiversified.csv"),
    weighted_output_csv=out_path("embeddings/mistral-weightedpooled-embeddings-300M-undiversified.csv"),
    sequence_col="sequence",
    batch_size=96,
)

