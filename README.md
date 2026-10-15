# ApexOracle DLM Pretraining

This repository preserves the collaborator-developed producer for the ApexOracle diffusion language model (DLM) and
its 209-descriptor multitask regression (MTR) objective. Downstream embedding, guidance, and candidate scoring belong
to [ApexOracle-MDLM](https://github.com/DragonDescentZerotsu/ApexOracle-MDLM), not this repository.

The implementation is derived from [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) and remains under
Apache-2.0. The original ApexOracle source is recoverable from the annotated tag
`legacy-code-snapshot-2026-08-10`; the clean branch intentionally limits changes to portability, validation, and
documentation.

## Environment

The historical training environment used CUDA 11.8, PyTorch 2.4.1, Lightning 2.2.1, Transformers 4.38.2, and
FlashAttention 2.6.3. A CUDA-capable Linux environment is required for the DiT training path.

```bash
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.6.3 --no-build-isolation
```

## Data contract

Training and validation inputs are Hugging Face datasets saved with `datasets.save_to_disk`. Each row must contain:

- `input_ids`: integer SELFIES tokens with shape `[1024]`;
- `attention_mask`: token mask with shape `[1024]`;
- `descriptors`: 209 finite RDKit descriptor values.

Set dataset locations explicitly; datasets and checkpoints are not distributed through Git:

```bash
export APEXORACLE_DLM_TRAIN_PATH=/path/to/prepared/train
export APEXORACLE_DLM_VALID_PATH=/path/to/prepared/validation
export APEXORACLE_DLM_CACHE_DIR=/path/to/cache
```

The IBM SELFIES tokenizer is fixed to revision `55e83392264cb998f7aa5014847df29868aefeb8`.
The bundled descriptor statistics are part of the model contract.

## Train DLM + MTR

The MTR head consumes 1024-dimensional hidden states, so the documented producer profile is `model=medium`:

```bash
python main.py \
  model=medium \
  data=SELFIES \
  wandb.name=dlm-mtr \
  parameterization=subs \
  model.length=1024 \
  sampling.steps=1000
```

This entry preserves the original loss, including the `0.1 × descriptor MSE` term. Dataset preparation and the
original large-scale checkpoint-producing run are not reconstructed by the synthetic smoke.

## Validation

CPU/source contracts:

```bash
python -m pytest -q
python -m compileall -q .
```

One-step CUDA backbone/MTR train-save-load smoke:

```bash
python scripts/smoke_train_save_load.py --device cuda:0
```

Historical source is available through the `legacy-code-snapshot-2026-08-10` tag.
The synthetic smoke validates one
shortened joint-objective training step, save and strict reload; it does not reproduce
the full pretraining run or its dataset.

## Acknowledgements

This repository was built from [MDLM](https://github.com/kuleshov-group/mdlm). The regression head notes its adaptation
from ChemBERTa's multitask regression implementation. See `NOTICE` and `LICENSE`.
