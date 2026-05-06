# Decisions Needed for Deferred NLP Candidates

The deterministic text/NLP atoms from `research/04_text_nlp_encoding.md` and
`research/04_research.pdf` were ingested into `sciona-atoms-ml` as
`sciona.atoms.ml.text_nlp`.

`g2p_convert` and the first tokenizer atom were originally deferred, but have
since been ingested with explicit artifact tooling. `back_translate` is closed
as not an atom.

## Resolved: `g2p_convert`

Status:

- Ingested as `sciona.atoms.ml.g2p`.
- Uses explicit artifact paths rather than runtime downloads or implicit cache
  lookups.
- Uses the vendored inference path needed for deterministic local execution,
  avoiding direct dependency on `g2p-en`'s declared GPL `Distance` dependency.
- Splits model/state from logic:
  - `checkpoint20.npz` for the neural G2P checkpoint.
  - `homographs.txt`, renamed from upstream `homographs.en` for artifact
    validation.
  - NLTK averaged perceptron English POS tagger as a multi-file JSON artifact
    directory.

Residual policy notes:

- The `g2p-en` wheel is Apache-labeled and contains the checkpoint, but does not
  provide a separate model-card or weight-specific license. This was treated as
  sufficient for the current ingestion, subject to later catalog policy.
- The direct `g2p-en` package install path remains unsuitable because it can
  trigger runtime NLTK downloads and declares the unused GPL `Distance`
  dependency.
- Detailed package, download, file-count, and minimal-asset findings are in
  [G2P.md](../G2P.md).

## Resolved: Transformer Tokenizer Wrappers

Status:

- Ingested a narrow first atom as `sciona.atoms.ml.tokenizer.tokenize`.
- Repo ownership is `sciona-atoms-ml`: tokenizers are preprocessing
  infrastructure that map text to token IDs and masks.
- Atom boundary wraps actual deterministic tokenization, not only
  post-processing over pre-computed IDs.
- Artifact shape is one `tokenizer.json` state port using the Hugging Face fast
  tokenizer single-file JSON format.
- Runtime dependency is `tokenizers`; `transformers` remains disallowed as a
  direct dependency for these atoms because it brings model loading and implicit
  download behavior.

Deferred expansion:

- Multi-file BPE or WordPiece assets such as `vocab.txt` and `merges.txt` can
  be added as a second state-port shape later.

SentencePiece `.model` support:

- Resolved via conversion rather than allowlisting. The `.model` format is
  protobuf binary — not pickle-dangerous, but adding it to the scanner
  would require a new loader policy and protobuf-aware validation.
- Instead, `scripts/convert_sentencepiece.py` in `sciona-atoms` converts
  `.model` files to HuggingFace `tokenizer.json` format using raw protobuf
  wire-format parsing (no `sentencepiece` wheel needed at conversion time).
- The resulting `tokenizer.json` passes the existing JSON format scanner
  and loads natively via `tokenizers.Tokenizer.from_file()`.

## Closed: `back_translate`

Decision:

- Do not ingest `back_translate` as an atom.
- A callable-only `back_translate(text, translation_fn)` boundary is an opaque
  wrapper around behavior outside the registry and matches the poisonous-atom
  pattern in `AGENT_INGESTION.md`.
- A real back-translation implementation would require large translation model
  artifacts per language pair, with complex provenance and tokenizer/model
  conventions.
- Back-translation is better represented as a data-augmentation CDG pipeline,
  not a reusable computational primitive.

Future shape:

- If needed later, model-backed translation should be represented as explicit
  forward-translation and reverse-translation atoms with their own local model
  artifacts. A back-translation workflow can then compose those atoms in a CDG.
