# G2P Runtime and Licensing Notes

This note captures the current findings for the uningested `g2p_convert`
candidate from `research/04_text_nlp_encoding.md` and
`research/04_research.pdf`.

The purpose is to preserve the observed package, model-weight, NLTK-data, and
purity concerns before Sciona gets an architectural path for model-backed or
asset-backed atoms.

## Candidate

`g2p_convert(text: str) -> list[str]`

The research candidate transduces English text into ARPAbet phonemes using:

- CMUDict dictionary lookup for known words
- POS tagging for homograph disambiguation
- bundled NumPy model weights for out-of-vocabulary prediction

## Package Findings

Checked package: `g2p-en==2.1.0`.

Findings:

- PyPI lists `g2p-en` as Apache Software License.
- The wheel contains `g2p_en-2.1.0.dist-info/LICENSE.txt`, which is Apache 2.0.
- The wheel includes `g2p_en/checkpoint20.npz`, so the model weights are shipped
  inside the Apache-labeled distribution artifact.
- There is no separate model-card or weight-specific license statement in the
  wheel.
- Package metadata declares dependencies on:
  - `numpy>=1.13.1`
  - `nltk>=3.2.4`
  - `inflect>=0.3.1`
  - `Distance>=0.1.3`

Licensing concerns:

- `Distance==0.1.3` is GPL-classified on PyPI.
- The downloaded `g2p-en` source path inspected here does not appear to import
  `distance`, but the declared dependency is still a packaging and licensing
  concern if installing `g2p-en` directly.
- `inflect>=0.3.1` is too loose. Historical `inflect 0.3.1` is AGPL-classified
  on PyPI; current `inflect` releases are MIT.
- CMUDict itself is BSD-style licensed.
- NLTK is Apache 2.0. NLTK data is distributed through NLTK's data mechanism,
  but any shipped asset set should carry explicit provenance for each required
  resource.

## Matcher Venv Changes Made During Investigation

Installed into `/Users/conrad/personal/sciona-matcher/.venv`:

- `nltk==3.9.4`
- `regex==2026.4.4` as an NLTK dependency
- `inflect==7.5.0`
- `typeguard==4.5.1` as an `inflect` dependency

`g2p-en` was not installed directly into the venv because its metadata declares
the GPL `Distance` dependency.

Instead:

- Downloaded `g2p_en-2.1.0-py3-none-any.whl`.
- Unpacked it to `/tmp/sciona-g2p-wheel-unpacked`.
- Ran it by adding that unpacked path to `sys.path`.

## Runtime Behavior

`g2p-en` imports NLTK and performs download checks at module import time:

- It checks for `taggers/averaged_perceptron_tagger.zip`.
- It checks for `corpora/cmudict.zip`.
- If missing, it calls `nltk.download(...)`.

That violates the atom purity boundary if used as-is. An atom must not perform
network downloads or hidden data installation during import or execution.

## First Isolated Run

The first run used:

- `NLTK_DATA=/tmp/sciona-g2p-nltk-run-data`
- `HOME=/tmp/sciona-g2p-nltk-run-home`

The initial import did not reach NLTK downloads because `inflect` was missing.
After installing current `inflect`, running the unpacked wheel downloaded:

```text
corpora/cmudict.zip
corpora/cmudict/README
corpora/cmudict/cmudict
taggers/averaged_perceptron_tagger.zip
taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle
```

That run then failed when executed directly from the zipped wheel because NumPy
could not load `checkpoint20.npz` from inside the zip archive. Running from the
unpacked wheel fixed the model-weight loading path.

## Current-NLTK Compatibility Issue

With the unpacked wheel and current NLTK, `G2p()` constructed, but calling it
failed because `nltk.pos_tag` now requires:

```text
taggers/averaged_perceptron_tagger_eng/
```

`g2p-en` 2.1.0 only checks/downloads the old resource:

```text
taggers/averaged_perceptron_tagger.zip
```

Manually downloading `averaged_perceptron_tagger_eng` added:

```text
taggers/averaged_perceptron_tagger_eng.zip
taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.classes.json
taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.tagdict.json
taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.weights.json
```

After that, the unpacked `g2p-en` wheel ran successfully.

Example outputs:

```text
I refuse to collect the refuse around here.
-> ['AY1', ' ', 'R', 'IH0', 'F', 'Y', 'UW1', 'Z', ' ', 'T', 'UW1', ' ', 'K', 'AH0', 'L', 'EH1', 'K', 'T', ' ', 'DH', 'AH0', ' ', 'R', 'EH1', 'F', 'Y', 'UW2', 'Z', ' ', 'ER0', 'AW1', 'N', 'D', ' ', 'HH', 'IY1', 'R', ' ', '.']

I'm an activationist.
-> ['AY1', 'M', ' ', 'AE1', 'N', ' ', 'AE2', 'K', 'T', 'IH0', 'V', 'EY1', 'SH', 'AH0', 'N', 'IH0', 'S', 'T', ' ', '.']
```

## Downloaded File Counts

Full working downloaded NLTK data set:

- 9 files
- about 20 MB

Full file list:

```text
corpora/cmudict.zip
corpora/cmudict/README
corpora/cmudict/cmudict
taggers/averaged_perceptron_tagger.zip
taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle
taggers/averaged_perceptron_tagger_eng.zip
taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.classes.json
taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.tagdict.json
taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.weights.json
```

Full set byte counts from the observed run:

```text
285       averaged_perceptron_tagger_eng.classes.json
3,808     corpora/cmudict/README
25,788    averaged_perceptron_tagger_eng.tagdict.json
896,069   corpora/cmudict.zip
1,539,115 averaged_perceptron_tagger_eng.zip
2,526,731 averaged_perceptron_tagger.zip
3,820,830 corpora/cmudict/cmudict
5,677,744 averaged_perceptron_tagger_eng.weights.json
6,138,625 averaged_perceptron_tagger.pickle
20,628,995 total
```

Additional bundled `g2p-en` assets:

- `g2p_en/checkpoint20.npz`: about 3.2 MB
- `g2p_en/homographs.en`: about 20 KB upstream; copy or rename to
  `homographs.txt` for Sciona artifact validation because `.en` is not an
  allowed serialized artifact extension.

## Minimal Offline Asset Set

A minimal offline run was tested with only these 3 NLTK zip files:

```text
corpora/cmudict.zip
taggers/averaged_perceptron_tagger.zip
taggers/averaged_perceptron_tagger_eng.zip
```

Size:

- 3 files
- about 4.7 MB

Byte counts:

```text
896,069   corpora/cmudict.zip
1,539,115 taggers/averaged_perceptron_tagger_eng.zip
2,526,731 taggers/averaged_perceptron_tagger.zip
4,961,915 total
```

This minimal set worked with the unmodified `g2p-en` 2.1.0 source and current
NLTK in the tested setup.

## Are All Files Necessary?

Necessary:

- `cmudict.zip`: yes. `G2p.__init__` calls `cmudict.dict()`.
- `averaged_perceptron_tagger_eng.zip`: yes with current NLTK.
  `nltk.pos_tag` loads this resource.
- `checkpoint20.npz`: yes for OOV neural prediction.
- `homographs.txt`: yes for homograph disambiguation. This is the upstream
  `homographs.en` content stored under an allowed text artifact extension.

Conditionally necessary:

- `averaged_perceptron_tagger.zip`: not needed by current NLTK `pos_tag`, but
  unmodified `g2p-en` checks for this old resource at import time and downloads
  it if missing. It is necessary only if we run unmodified `g2p-en`.

Not necessary if zip files are retained:

- Extracted `corpora/cmudict/*`
- Extracted `taggers/averaged_perceptron_tagger/*`
- Extracted `taggers/averaged_perceptron_tagger_eng/*`

Those are downloader byproducts. NLTK was able to run from the zip files in the
tested setup.

## Architectural Implications for Sciona

This candidate is a good example of an asset-backed atom, not a normal pure
function atom.

A future architecture should probably separate:

- pure atom interface
- immutable asset manifest
- asset provenance and licenses
- asset resolution/loading
- network/download prohibition
- optional model-weight trust policy
- package dependency policy

For `g2p_convert`, a viable Sciona-native design would likely:

- avoid direct `pip install g2p-en` unless the GPL `Distance` dependency is
  resolved
- vendor or reimplement only the needed inference path
- require pre-resolved asset paths or explicit asset objects
- include an asset manifest for:
  - CMUDict
  - current NLTK English perceptron tagger
  - old NLTK perceptron tagger only if unmodified `g2p-en` is used
  - `checkpoint20.npz`
  - `homographs.txt`
- fail fast if assets are missing, without downloading
- record separate provenance for package source, dictionary, tagger, and model
  weights

## Current Recommendation

Do not ingest `g2p_convert` as a standard atom yet.

Wait for the architectural change that can represent immutable external assets
and model weights explicitly. Then ingest either:

1. a dictionary-only G2P atom in `sciona-atoms-ml`, or
2. a model-backed G2P atom in `sciona-atoms-dl` with an asset manifest and no
   runtime downloads.
