# Portability and behavior audit

> Audit date: 2026-08-10

## Verified facts

- The original `models/__init__.py` imported an absent `ditfb` module and could not import the model package.
- Descriptor statistics were loaded from one author's absolute filesystem path even though byte-identical arrays were
  already tracked at repository root.
- The SELFIES tokenizer cache and train/validation dataset directories were fixed to author-machine paths.
- The training constructor loaded an unrelated generative-perplexity tokenizer even when that evaluation was disabled.
- The MTR head input is fixed at 1024 and has 209 outputs. `configs/model/medium.yaml` uses hidden size 1024, while the
  original README selected the 768-hidden `small` profile.
- Descriptor arrays both have shape `[209]`, contain only finite values, and retain their original bytes:
  - `descriptors_mean.npy`: `19d2693e9797f848146dc27536c1861cd331008dc9fe9e003b168ad6210b5e93`
  - `descriptors_std.npy`: `c69b149f0f561b4ee7669b0e3cef82fbaea2fa630142a8571fa31f0e8b6bf608`

## Minimal clean-branch changes

- Removed only the missing `ditfb` import; no model parameter was removed.
- Resolved the existing descriptor arrays relative to `models/dit.py`; they remain ordinary non-persistent tensors, so
  the historical model state-dict key schema is unchanged.
- Added explicit SELFIES train/validation/cache configuration and a fixed tokenizer revision.
- Made generative-perplexity tokenizer/model loading conditional on the existing evaluation flag.
- Corrected the documented producer profile to `model=medium`; the 1024-input MTR head itself is unchanged.
- Replaced author-machine cache and evaluator-checkpoint defaults in inherited upstream examples with explicit
  environment inputs; the example algorithms and Hydra overrides are unchanged.
- Added source contracts and a CUDA synthetic joint-objective train/save/load smoke.

## Evidence boundary

The synthetic smoke validates construction, one joint token/descriptor optimizer step, serialization, strict reload,
and deterministic post-reload outputs for a shortened one-block 1024-hidden model. It does not reproduce the full
24-block training run, the private curated dataset, the formal optimizer trajectory, or a historical checkpoint.

The generic upstream text configs and shell launchers are retained for source provenance. Their cache/checkpoint
locations are now explicit environment inputs and remain outside the canonical `data=SELFIES` entry.

## Synthetic runtime result

The canonical smoke passed on an NVIDIA H100 with PyTorch 2.3.1, CUDA 12.1 runtime, Lightning 2.2.1, and
FlashAttention 2.6.3. The validated synthetic profile used batch size 2, sequence length 8, one DiT block, hidden size
1024, and 209 descriptor outputs. The joint loss was finite; the saved full Diffusion state loaded with `strict=True`,
descriptor statistics remained absent from checkpoint keys, and post-reload backbone/MTR outputs were exactly equal.
