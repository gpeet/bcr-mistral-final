---
license: other
license_name: all-rights-reserved
license_link: LICENSE
gated: true
extra_gated_heading: Access to the BCR-Mistral weights
extra_gated_prompt: >-
  These model weights are not openly licensed. They are shared so that the results in the
  accompanying paper can be examined and verified. By requesting access you confirm that you will
  not redistribute or modify them, or use them for any other purpose, without written permission.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
extra_gated_button_content: Request access
library_name: transformers
pipeline_tag: feature-extraction
tags:
  - biology
  - immunology
  - B-cell-receptor
  - antibody
  - multiple-sclerosis
  - mistral
---

# BCR-Mistral

Decoder-only (Mistral-architecture) language models pretrained on human B cell receptor heavy-chain
(IGH) nucleotide sequences, from:

> Peet GC, Owens GP, Bennett JL, Krishnan A, Macklin WB. *Language Model Embedding Classifiers Enable
> Identification of Multiple Sclerosis-Associated BCRs and Repertoires.* (citation to be added)

Code and figure reproduction: https://github.com/gpeet/bcr-mistral-final
Processed data and TRUST4 assemblies: Zenodo, doi:10.5281/zenodo.20432913

`scripts/fetch_data.sh models` in the repository downloads this repo into the layout the pipeline
expects. `SHA256SUMS` covers every file.

## Licence

These weights are **not** openly licensed. Copyright 2026 The Regents of the University of Colorado;
all rights reserved. No copyright or patent licence is granted. You may download and run them to
examine or verify the results in the paper; any other use, including redistribution, modification and
commercial use, needs written permission. See `LICENSE`, and contact Graham Peet
(graham.peet@cuanschutz.edu).

The processed data record (doi:10.5281/zenodo.20432913) is licensed separately under CC BY 4.0.

## Models

| File | Earlier label | Parameters | Architecture (hidden / layers / heads / KV heads / MLP) | Training data |
|---|---|---|---|---|
| `2025-HL-Foundation-1024x4096-HF/` and `2025-HL-Foundation-1024x4096-gradClip-long-cp48h.pth` | 500M (used for all analyses) | 487 M | 1024 / 32 / 32 / 8 / 4096 | diversity-selected IGH, 48 h checkpoint |
| `2025-HL-Foundation-300M-gradClip.pth` | 300M | 151 M | 768 / 16 / 12 / 12 / 3072 | diversity-selected IGH |
| `2025-HL-Foundation-100M-gradClip.pth` | 100M | 50 M | 512 / 12 / 8 / 8 / 2048 | diversity-selected IGH |
| `2025-HL-Foundation-300M-gradClip-undiversified.pth` | 300M, random sample | 151 M | as 300M | size-matched random IGH sample (ablation) |
| `2026-Foundation-300M-gradClip-protein.pth` | 300M, protein | 151 M | as 300M | the diverse corpus translated to amino acids |
| `2025-HL-Foundation-1024x4096-gradClip-junction.pth` | junction model | 487 M | as 500M | CDR3 junction sequences |

`trained/sequence-classifier/` holds the downstream models from the paper: the four sequence-classifier
ensembles (10 models per outer fold; full-sequence and junction embeddings, with and without label
exclusion) and the two waypoint models with the patient feature tables computed from them. The code
repository reuses them by default, so the figures regenerate exactly.

All foundation models use a 512-token context (the junction model: 96). Nucleotide models use
`tokenizers/base_only_bpe_tokenizer`; the protein model uses `tokenizers/protein_bpe_tokenizer`.
`SHA256SUMS` lists checksums for every file.

**Pretraining data.** IGH sequences from the Observed Antibody Space (OPIG OAS, unpaired), sampled for
diversity in 7-mer space. During pretraining each sequence was cyclically rotated and given zero to two
masked blocks of 20 tokens, and truncated at the context length. Trained with ROCm 5.2.3 on AMD MI100
GPUs (CU Boulder Alpine). See the paper's Methods.

## Usage: sequence embeddings

The paper's classifiers use the final-layer hidden state of the last token, with left padding:

```python
import torch
from transformers import MistralForCausalLM, PreTrainedTokenizerFast

tok = PreTrainedTokenizerFast(tokenizer_file="tokenizers/base_only_bpe_tokenizer/tokenizer.json",
                              padding_side="left")
tok.add_special_tokens({"pad_token": "[PAD]"})
# from a local download; from the Hub use
#   MistralForCausalLM.from_pretrained("grpeet/bcr-mistral",
#                                      subfolder="2025-HL-Foundation-1024x4096-HF")
model = MistralForCausalLM.from_pretrained("2025-HL-Foundation-1024x4096-HF").eval()

seqs = ["GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCC..."]
enc = tok(seqs, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
with torch.no_grad():
    out = model(**enc, output_hidden_states=True)
embedding = out.hidden_states[-1][:, -1, :]      # (n_sequences, 1024)
```

The other models are PyTorch state dicts: build `MistralConfig` with the architecture above (the training
scripts in the code repository give the exact configs) and call `model.load_state_dict(torch.load(path))`.

## Intended use and limitations

Research use: embedding and scoring BCR heavy-chain sequences. The models were trained on human IGH
only; behaviour on light chains, TCRs or other species is untested. The MS sequence and patient
classifiers built on these embeddings are research tools, not diagnostics.

