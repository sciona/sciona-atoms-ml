# Decisions Needed for Uningested NLP Candidates

The deterministic text/NLP atoms from `research/04_text_nlp_encoding.md` and
`research/04_research.pdf` were ingested into `sciona-atoms-ml` as
`sciona.atoms.ml.text_nlp`.

The following candidates were intentionally left uningested because they need
repo-boundary, dependency, or licensing decisions before becoming registry
atoms.

## Transformer Tokenizer Wrappers

Decision points:

- Repo ownership: likely `sciona-atoms-dl`, because these are transformer-related
  rather than classical text features.
- Atom boundary: decide whether the catalog should expose a wrapper around
  HuggingFace `tokenizers` / `AutoTokenizer`, or only pure offset and array
  post-processing helpers.
- Dependency surface: adding `tokenizers` is relatively lightweight; adding
  `transformers` is heavier and brings model/config loading behavior into the
  provider repo.
- Purity boundary: runtime downloads and implicit model lookup must be
  disallowed. Any tokenizer object or vocab/config should be explicit or loaded
  deterministically outside the atom.

Recommendation:

- Ingest tokenizer offset and encoding wrappers only if `tokenizers` is approved
  as a provider dependency and the atom accepts explicit tokenizer state or a
  deterministic local tokenizer object.

## `g2p_convert`

Decision points:

- Repo ownership: `sciona-atoms-dl` if treated as model-backed sequence
  transduction; possibly `sciona-atoms-ml` if limited to dictionary lookup.
- Atom boundary: dictionary-only CMU lookup vs model-backed out-of-vocabulary
  prediction.
- Dependency and licensing: packages such as `g2p_en`, `nltk`, pronunciation
  dictionaries, and bundled model weights need license review before ingestion.
- Purity boundary: model weights or dictionaries must be immutable local assets;
  no runtime downloads or hidden file/network lookups inside the atom.

Recommendation:

- Defer model-backed `g2p_convert` until package, dictionary, and model-weight
  licensing is confirmed. A dictionary-only atom may be acceptable sooner if the
  dictionary license is compatible and the dictionary is passed explicitly.

Research notes:

- `g2p-en` 2.1.0 is listed on PyPI as Apache Software License and its wheel
  contains an Apache 2.0 `LICENSE.txt`.
- The `g2p-en` wheel includes `g2p_en/checkpoint20.npz`, so the model weights
  are distributed in the same Apache-labeled artifact. There is no separate
  model-card or weight-specific license in the wheel.
- The package metadata declares dependencies on `numpy`, `nltk`, `inflect`, and
  `Distance`. `Distance==0.1.3` is GPL-classified on PyPI. The downloaded
  `g2p-en` source does not appear to import `distance`, but the declared
  dependency is still a packaging/licensing concern if installing `g2p-en`
  directly.
- `inflect>=0.3.1` is the declared lower bound in `g2p-en`. Historical
  `inflect` 0.3.1 is AGPL-classified on PyPI, while current `inflect` releases
  are MIT. Any dependency approval should pin a permissive `inflect` version
  rather than accepting the old lower bound.
- `g2p-en` imports NLTK `pos_tag` and `cmudict`, and calls `nltk.download(...)`
  at import time when `averaged_perceptron_tagger` or `cmudict` is missing.
  That violates the atom purity boundary unless data assets are pre-provisioned
  and downloads are disabled.
- CMUDict itself is under a BSD-style license. NLTK and the NLTK data repository
  are Apache 2.0, but the atom should still carry explicit provenance for the
  dictionary and POS tagger assets if those assets are shipped or required.
- Detailed package, download, file-count, and minimal-asset findings are in
  [G2P.md](G2P.md).

Updated decision points:

- Decide whether direct `g2p-en` installation is acceptable despite the declared
  GPL `Distance` dependency and historical AGPL-compatible lower bound on
  `inflect`.
- Decide whether to vendor or reimplement only the needed G2P inference path so
  the atom avoids unused GPL dependencies and import-time downloads.
- Decide whether model weights from an Apache-labeled wheel are sufficient
  provenance, or whether the catalog requires an explicit model-weight license
  statement separate from package metadata.
- Decide whether the atom can require pre-provisioned NLTK data assets, or
  whether all dictionaries/taggers must be passed explicitly as inputs.

## `back_translate`

Decision points:

- Repo ownership: likely `sciona-atoms-dl`.
- Atom boundary: decide whether accepting an explicit deterministic
  `translation_fn` is a valid atom, or whether it is too opaque because the
  actual model behavior lives outside the atom.
- Purity boundary: cloud/API translation must be excluded. Only local,
  deterministic translation callables should be considered.
- Provenance: decide whether the atom is a real implementation or only an
  orchestration boundary before citing back-translation sources.

Recommendation:

- Defer `back_translate` until the atom catalog has a clear policy for callable
  model boundaries and local model-weight provenance.
