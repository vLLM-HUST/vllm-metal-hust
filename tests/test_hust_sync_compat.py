# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression for the upstream merge and HUST mirror compatibility.

Run independently of the Metal/torch pytest fixtures:
python -m unittest discover -s tests -p test_hust_sync_compat.py
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def load_compat():
    spec = importlib.util.spec_from_file_location(
        "compat_under_test", Path(__file__).parents[1] / "vllm_metal/compat.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyncCompatibilityTests(unittest.TestCase):
    def test_registration_keeps_hust_and_new_upstream_patches(self):
        compat = load_compat()
        names = [
            "_patch_huggingface_hub_relative_redirect_query",
            "_patch_torch_mps_empty_host_cache",
            "_patch_vllm_gemma4_mtp_config_loading",
            "_apply_bytelevel_patch_during_registration",
            "_patch_mlx_lm_qwen35_fp8_sanitize",
            "_patch_mlx_lm_qwen3_flat_weight_prefix",
            "_patch_transformers_exaone4_config",
        ]
        calls = []
        for name in names:
            setattr(compat, name, lambda n=name: calls.append(n))
        compat.apply_compat_patches()
        compat.apply_compat_patches()
        self.assertEqual(calls, names)

    def test_relative_mirror_redirect_retains_query_and_patch_is_idempotent(self):
        compat = load_compat()
        calls = []
        responses = iter(
            [
                SimpleNamespace(
                    status_code=302,
                    headers={"Location": "/cache/file?etag=123&size=456"},
                ),
                SimpleNamespace(status_code=200, headers={"ETag": "123"}),
            ]
        )

        class Client:
            def __init__(self, *, trust_env):
                self.trust_env = trust_env

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def request(self, **kwargs):
                calls.append(kwargs)
                return next(responses)

        http = SimpleNamespace(
            _httpx_follow_relative_redirects_with_backoff=lambda *a, **k: None,
            hf_raise_for_status=lambda response: None,
        )
        download = SimpleNamespace()
        hub, utils = ModuleType("huggingface_hub"), ModuleType("huggingface_hub.utils")
        hub.file_download = download
        utils._http = http
        with patch.dict(
            sys.modules,
            {
                "httpx": SimpleNamespace(Client=Client),
                "huggingface_hub": hub,
                "huggingface_hub.utils": utils,
            },
        ):
            compat._patch_huggingface_hub_relative_redirect_query()
            wrapped = http._httpx_follow_relative_redirects_with_backoff
            compat._patch_huggingface_hub_relative_redirect_query()
            self.assertIs(wrapped, http._httpx_follow_relative_redirects_with_backoff)
            result = wrapped("HEAD", "https://mirror.example/model/file")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(
            calls[1]["url"], "https://mirror.example/cache/file?etag=123&size=456"
        )
        self.assertIs(download._httpx_follow_relative_redirects_with_backoff, wrapped)

    def test_mps_empty_host_cache_guard_preserves_other_accelerators(self):
        compat = load_compat()
        calls = []
        accelerator = SimpleNamespace(
            current_accelerator=lambda: SimpleNamespace(type="mps"),
            empty_host_cache=lambda: calls.append("original"),
        )
        torch = SimpleNamespace(accelerator=accelerator)
        with patch.dict(sys.modules, {"torch": torch}):
            compat._patch_torch_mps_empty_host_cache()
            guarded = accelerator.empty_host_cache
            compat._patch_torch_mps_empty_host_cache()
            self.assertIs(guarded, accelerator.empty_host_cache)

            guarded()
            accelerator.current_accelerator = lambda: SimpleNamespace(type="cpu")
            guarded()

        self.assertEqual(calls, ["original"])


if __name__ == "__main__":
    unittest.main()
