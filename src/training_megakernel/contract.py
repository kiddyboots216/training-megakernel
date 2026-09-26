"""Execution contract every build of the resident training program shares.

The decoder depth and tokens per GPU are chosen per build; they come from each
bundle's launch ABI (``geometry.Geometry``), not from this module.
"""

VOCAB = 151_936
PHYSICAL_SLOTS = 2
WORLD_SIZE = 8

PROGRAM_CTAS = 132
PROGRAM_THREADS = 384
