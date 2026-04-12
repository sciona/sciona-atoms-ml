"""ML-owned atom providers.

`datadriven` was inspected but left out of the first migration pass because it
mixes model-discovery and general numerics concerns, so its ownership is not
clear-cut enough to move without a broader classification decision.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
