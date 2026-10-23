#!/bin/bash
set -euo pipefail

# Install the pinned Python packages (envs/requirements-rocm.txt) into a pip --target
# directory, BCRM_PYLIBS, that sits on PYTHONPATH next to an existing PyTorch 1.13.0.
#
# On Alpine, torch comes from CURC's ROCm module and this must run on a GPU node
# (workflow/run.sh env-python does both). Elsewhere, set BCRM_INSTALL_TORCH=1 to also
# install torch 1.13.0 (ROCm 5.2 wheel; set BCRM_TORCH_INDEX for a CUDA build instead).
#
# Environment:
#   BCRM_PYLIBS          install directory (required)
#   BCRM_INSTALL_TORCH   1 to install torch into BCRM_PYLIBS too (default: 0)
#   BCRM_TORCH_INDEX     wheel index for torch (default: https://download.pytorch.org/whl/rocm5.2)

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${BCRM_PYLIBS:?set BCRM_PYLIBS to the install directory}"
export PYTHONNOUSERSITE=1

mkdir -p "$BCRM_PYLIBS"
if [ "${BCRM_INSTALL_TORCH:-0}" = 1 ]; then
    pip install --no-user --no-deps --target "$BCRM_PYLIBS" torch==1.13.0 \
        --index-url "${BCRM_TORCH_INDEX:-https://download.pytorch.org/whl/rocm5.2}"
    export PYTHONPATH="$BCRM_PYLIBS:${PYTHONPATH:-}"
fi

python -c "import torch; print('torch', torch.__version__)"
# --no-deps: the requirements file is the complete set; letting pip resolve would
# install a different PyTorch next to the one on PYTHONPATH
pip install --no-user --no-deps --target "$BCRM_PYLIBS" -r "$REPO_ROOT/envs/requirements-rocm.txt"
PYTHONPATH="$BCRM_PYLIBS:${PYTHONPATH:-}" python - <<'PY2'
import torch, transformers, optuna, pandas, sklearn
from prokbert.prokbert_tokenizer import ProkBERTTokenizer   # diversity selection
print("torch", torch.__version__, "| transformers", transformers.__version__, "| optuna", optuna.__version__)
PY2

# Record exactly what was installed
pip freeze --path "$BCRM_PYLIBS" > "$BCRM_PYLIBS/installed.txt"
echo "Installed into $BCRM_PYLIBS (see installed.txt)"
