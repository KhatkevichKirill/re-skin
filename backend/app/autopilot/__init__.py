"""The "by Assistant" autopilot (AICP-363, docs/autopilot-design.md).

Kept import-light: ``app.spend`` imports ``app.autopilot.spend_view`` on its
hot path. Import the submodules you need (contract, registry, core, actions).
"""
