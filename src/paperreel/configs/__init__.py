"""Bundled YAML configs loaded via ``importlib.resources``.

Anything in this package directory ending in ``.yaml`` is shippable
with the wheel and discoverable via :func:`paperreel.config.load_config`
by its base name (``bigvram`` -> ``bigvram.yaml``).
"""
