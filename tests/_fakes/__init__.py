"""Test-only fake providers.

These are NOT shipped to users. The production build requires real local
backends (Ollama / Coqui XTTS / SDXL); these fakes exist so unit tests
can exercise the full pipeline shape without weights, GPU, or network.
"""
