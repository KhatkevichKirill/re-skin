"""Fal endpoint registry (AICP-312 Stage 1, docs/fal-provider-design.md §3).

Fail closed: an unknown endpoint, or a resolution without a tariff, RAISES —
it never falls back to a guess, because the tariff is what the daily cap
reserves. Stage 1 has the reframe model AICP-251 needs; AICP-358 adds the VOID
inpaint model for subtitle-removal caption fills (a flat price per request, and a
result checked by frame count — VOID tags its output 12 fps whatever the input).

Tariffs are docs-only (the pricing endpoint returns only the base rate), in USD
per OUTPUT second. The overlap factor is the worst case for the billable overlap
windows Fal adds on long sources; 2.0 is unmeasured (design *Known residuals*).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


class UnknownEndpoint(ValueError):
    pass


class UntariffedResolution(ValueError):
    pass


@dataclass(frozen=True)
class FalEndpoint:
    endpoint_id: str
    family: str                       # spend kind is "fal_<family>"
    tariff_usd_per_sec: dict[str, float]
    overlap_factor: float
    min_input_sec: float
    max_input_sec: float
    aspect_ratios: frozenset[str]
    # Parameters the frozen spec may carry; anything else is refused.
    allowed_params: frozenset[str] = field(default_factory=frozenset)
    # AICP-358. A flat price per request replaces the per-second tariff when set
    # (worst_case_usd returns it as is; usd_per_sec still guards the resolution).
    flat_usd: Optional[float] = None
    # "duration": the result must last output_sec (reframe); "frames": it must
    # hold exactly the frame count frozen at admission (VOID).
    media_check: str = "duration"
    # fal_submit sends the task's resolution unless the endpoint rejects it.
    send_resolution: bool = True
    # Input params every task of this endpoint must carry, each exactly once.
    required_inputs: frozenset[str] = field(default_factory=frozenset)

    @property
    def queue_app(self) -> str:
        """owner/app = the FIRST TWO path segments; status/result/cancel live
        under it (the full endpoint path answers 405 there)."""
        return "/".join(self.endpoint_id.split("/")[:2])

    def usd_per_sec(self, resolution: str) -> float:
        rate = self.tariff_usd_per_sec.get(resolution)
        if rate is None:
            raise UntariffedResolution(
                f"{self.endpoint_id}: no tariff for resolution {resolution!r}"
            )
        return rate

    def worst_case_usd(self, resolution: str, output_sec: float) -> float:
        if self.flat_usd is not None:
            self.usd_per_sec(resolution)  # still fail closed on the resolution
            return round(float(self.flat_usd), 4)
        sec = max(1.0, math.ceil(float(output_sec)))
        return round(sec * self.usd_per_sec(resolution) * self.overlap_factor, 4)


FAL_ENDPOINTS: dict[str, FalEndpoint] = {
    "minimax/h3-max/reframe-video": FalEndpoint(
        endpoint_id="minimax/h3-max/reframe-video",
        family="reframe",
        # 1080p is listed by the schema but has "no rate listed": refused.
        tariff_usd_per_sec={"480p": 0.05, "768p": 0.08},
        overlap_factor=2.0,
        min_input_sec=0.25,
        max_input_sec=300.0,
        aspect_ratios=frozenset({"16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"}),
        allowed_params=frozenset({"aspect_ratio", "resolution", "seed"}),
    ),
    # AICP-358: masked caption fill. flat_usd is the RESERVATION: the real charge
    # is $0.05 per request (Fal logs, two jobs on run 2845eef3, 2026-10-31); the
    # reserve keeps 2x headroom (= the top of Fal's listed $0.05-0.10).
    # The tariff dict only names the one input resolution the fill sends.
    "fal-ai/void-video-inpainting": FalEndpoint(
        endpoint_id="fal-ai/void-video-inpainting",
        family="inpaint",
        tariff_usd_per_sec={"720p": 0.0},
        overlap_factor=1.0,
        min_input_sec=2.3,
        max_input_sec=60.0,
        aspect_ratios=frozenset(),
        allowed_params=frozenset({"prompt", "num_frames", "enable_pass2_refinement"}),
        flat_usd=0.10,
        media_check="frames",
        send_resolution=False,
        required_inputs=frozenset({"video_url", "quad_mask_video_url"}),
    ),
}

VOID_ENDPOINT = "fal-ai/void-video-inpainting"


def endpoint(endpoint_id: str) -> FalEndpoint:
    spec = FAL_ENDPOINTS.get(endpoint_id)
    if spec is None:
        raise UnknownEndpoint(f"unknown Fal endpoint {endpoint_id!r}")
    return spec
