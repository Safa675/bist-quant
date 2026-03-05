"""Legacy batch signal runner placeholder.

Historically this module executed a full integration sweep at import time,
which is not suitable for automated CI runs.
"""

from __future__ import annotations

import pytest

pytest.skip(
    "Legacy manual signal sweep is intentionally excluded from automated pytest.",
    allow_module_level=True,
)
