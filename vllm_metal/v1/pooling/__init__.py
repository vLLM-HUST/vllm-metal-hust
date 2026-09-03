# SPDX-License-Identifier: Apache-2.0
"""Pooling backend seam used by ``MetalModelRunner``.

Start at ``model_runner.py`` for scheduling and output attachment.
Then read ``contract.py`` for the DTOs/protocols, ``backends/decoder/runtime.py``
for current decoder pooling, and model-family files only for task-specific
behavior such as Qwen3 reranker scoring.
"""
