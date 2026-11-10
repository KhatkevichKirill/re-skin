"""Production autopilot flows register themselves here on import.

AICP-364 (subtitle removal) and AICP-365 / AICP-366 (localisation) add theirs:
import the module below so that it calls ``app.autopilot.registry.register(flow)``.
"""

from . import subtitle_removal  # noqa: E402,F401  (AICP-364: registers the flow)
from . import localisation  # noqa: E402,F401  (AICP-365: localisation generation, C1)
from . import localisation_finish  # noqa: E402,F401  (AICP-366: QA and finishing, C2)
from . import localisation_finish_handlers  # noqa: E402,F401  (registers the C2 handlers)
