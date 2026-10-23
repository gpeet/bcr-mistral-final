"""
Write a BCR-Mistral checkpoint as a Hugging Face model folder (config.json + model.safetensors),
so it loads with MistralForCausalLM.from_pretrained().

Used to package the released 500M model, the 48 h checkpoint the paper uses throughout:
    python foundation-models/export_hf.py \\
        --config  $BCRM_MODELS/2025-HL-Foundation-1024x4096-HF \\
        --weights $BCRM_MODELS/2025-HL-Foundation-1024x4096-gradClip-long-cp48h.pth \\
        --out     <release>/huggingface/2025-HL-Foundation-1024x4096-HF
"""

import argparse

import torch
from transformers import MistralConfig, MistralForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",  required=True, help="Folder with the model's config.json")
    parser.add_argument("--weights", required=True, help=".pth state dict")
    parser.add_argument("--out",     required=True, help="Output folder")
    args = parser.parse_args()

    model = MistralForCausalLM(MistralConfig.from_pretrained(args.config))
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.save_pretrained(args.out, safe_serialization=True)

    # The saved folder must give the same model back
    reloaded = MistralForCausalLM.from_pretrained(args.out)
    for k, v in model.state_dict().items():
        assert torch.equal(v, reloaded.state_dict()[k]), k
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
