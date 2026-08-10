#!/usr/bin/env python3
"""Run a small CUDA DLM+MTR optimization and strict save/load smoke."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from diffusion import Diffusion


class SyntheticTokenizer:
  mask_token = '[MASK]'
  bos_token_id = 0
  eos_token_id = 1

  def __init__(self, vocab_size: int):
    self.vocab_size = vocab_size
    self.mask_token_id = vocab_size - 1

  def get_vocab(self):
    return {str(index): index for index in range(self.vocab_size)}


def build_config(sequence_length: int, blocks: int):
  return OmegaConf.create({
    'backbone': 'dit',
    'parameterization': 'subs',
    'T': 0,
    'subs_masking': False,
    'time_conditioning': True,
    'model': {
      'hidden_size': 1024,
      'cond_dim': 128,
      'length': sequence_length,
      'n_blocks': blocks,
      'n_heads': 16,
      'scale_by_sigma': True,
      'dropout': 0.0,
    },
    'sampling': {'predictor': 'ddpm_cache'},
    'eval': {
      'gen_ppl_eval_model_name_or_path': 'gpt2-large',
      'compute_generative_perplexity': False,
    },
    'training': {
      'antithetic_sampling': True,
      'importance_sampling': False,
      'change_of_variables': False,
      'sampling_eps': 1e-3,
      'ema': 0.0,
    },
    'noise': {'type': 'loglinear'},
    'optim': {'lr': 1e-4},
    'data': {'cache_dir': './.cache'},
  })


def joint_loss(model, input_ids, sigma):
  del sigma
  attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
  descriptor_targets = torch.zeros(
    input_ids.shape[0], 209, device=input_ids.device, dtype=torch.float32)
  losses = model._loss(input_ids, attention_mask, descriptor_targets)
  return losses.loss + 0.1 * losses.reg_mse, losses.loss, losses.reg_mse


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--batch-size', type=int, default=2)
  parser.add_argument('--sequence-length', type=int, default=8)
  parser.add_argument('--blocks', type=int, default=1)
  parser.add_argument('--vocab-size', type=int, default=32)
  return parser.parse_args()


def main():
  args = parse_args()
  if not torch.cuda.is_available() or not args.device.startswith('cuda'):
    raise RuntimeError('The FlashAttention producer smoke requires a CUDA device.')

  torch.manual_seed(20260810)
  device = torch.device(args.device)
  config = build_config(args.sequence_length, args.blocks)
  tokenizer = SyntheticTokenizer(args.vocab_size)
  model = Diffusion(config, tokenizer=tokenizer).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  input_ids = torch.randint(
    0, args.vocab_size, (args.batch_size, args.sequence_length), device=device)
  sigma = torch.full((args.batch_size,), 0.5, device=device)

  model.train()
  optimizer.zero_grad(set_to_none=True)
  loss, token_loss, descriptor_loss = joint_loss(model, input_ids, sigma)
  if not torch.isfinite(loss):
    raise RuntimeError('Synthetic joint loss is not finite.')
  loss.backward()
  optimizer.step()

  state = model.state_dict()
  if any(key.endswith(('mean_stats', 'std_stats')) for key in state):
    raise RuntimeError('Descriptor statistics unexpectedly changed the checkpoint key schema.')

  with tempfile.TemporaryDirectory(prefix='apexoracle-dlm-smoke-') as temporary:
    checkpoint_path = Path(temporary) / 'synthetic.pt'
    torch.save({
      'model': state,
      'optimizer': optimizer.state_dict(),
      'config': OmegaConf.to_container(config, resolve=True),
    }, checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded = Diffusion(
      OmegaConf.create(payload['config']),
      tokenizer=SyntheticTokenizer(args.vocab_size)).to(device)
    reloaded.load_state_dict(payload['model'], strict=True)

  model.eval()
  reloaded.eval()
  with torch.no_grad():
    expected = model.backbone(input_ids, sigma, regress=True)
    actual = reloaded.backbone(input_ids, sigma, regress=True)
  for expected_tensor, actual_tensor in zip(expected, actual):
    if not torch.equal(expected_tensor, actual_tensor):
      raise RuntimeError('Post-reload model outputs are not exact.')

  print(json.dumps({
    'status': 'passed',
    'device': str(device),
    'batch_size': args.batch_size,
    'sequence_length': args.sequence_length,
    'blocks': args.blocks,
    'hidden_size': 1024,
    'descriptor_outputs': 209,
    'token_loss': float(token_loss.detach().cpu()),
    'descriptor_loss': float(descriptor_loss.detach().cpu()),
    'joint_loss': float(loss.detach().cpu()),
  }, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
