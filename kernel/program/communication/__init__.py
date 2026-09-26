"""The program's communication between the eight GPUs.

``memory_ops`` has the inline-PTX loads, stores and multicast operations the collectives are
written with, and the step loop's exit branch. ``symmetric_memory`` allocates the build's stand-ins
for the symmetric-memory arenas, which ``training_megakernel.arenas`` lays out; ``embedding_route``
moves embedding rows and
their gradients
rank to rank. ``decoder_fabric`` and ``shell_fabric`` build the kernel arguments of the ABI groups
``fabric`` (the decoder's weight all-gather and gradient reduce-scatter) and ``full_shell`` (the
shell's communication). The collectives themselves run in ``training_program``.
"""
