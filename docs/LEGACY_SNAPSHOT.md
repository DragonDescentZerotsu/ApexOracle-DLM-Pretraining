# Legacy source snapshot

## Verified recovery points

- Source repository and path: `DragonDescentZerotsu/ApexOracle`, legacy `DLM_pretrain/` subtree.
- Original monorepo commit: `2f29dee9cf6b7750425414f66c1a2d67998cb87f`.
- History-preserving subtree root: `fda167cf5fb90ac57952482fb5c0e605b188c105`.
- Recovery branch: `legacy/apexoracle-dlm-pretraining-2026-08-10`.
- Annotated recovery tag: `legacy-code-snapshot-2026-08-10`.
- Original tracked tree: 56 files, approximately 324 KiB; no checkpoint or dataset was present.

`manifests/legacy_source.sha256` records SHA-256 for every original tracked file. The recovery branch and tag are the
authority for exact source restoration; the manifest is an independent integrity check.

## Scientific boundary

The snapshot contains the collaborator-developed modification of upstream MDLM that adds a 209-output descriptor
regression head and combines its MSE with the diffusion token loss. It also contains generic upstream text-dataset
configs and launchers. Those inherited examples are retained for provenance but are not ApexOracle's canonical
SELFIES producer entry.

The snapshot does not contain the curated 121.6M-sequence dataset, data-preparation pipeline, training logs, or the
formal checkpoint. Consequently it identifies a verified historical producer family, not a fully closed
commit-to-checkpoint reconstruction.
