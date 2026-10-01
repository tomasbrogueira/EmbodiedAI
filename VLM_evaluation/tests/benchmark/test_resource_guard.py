"""A resident module must never be loaded over by a cache-clearing adapter."""

import unittest

from traversability_benchmark.profiling import _require_idle_kernel


class ResourceGuardTests(unittest.TestCase):
    def test_other_process_device_usage_does_not_block_idle_kernel(self):
        _require_idle_kernel({"allocated_bytes": 0, "reserved_bytes": 0, "device_used_bytes": 8_000_000_000})

    def test_resident_allocations_or_cached_blocks_require_dedicated_kernel(self):
        for baseline in ({"allocated_bytes": 1, "reserved_bytes": 1}, {"allocated_bytes": 0, "reserved_bytes": 1}):
            with self.subTest(baseline=baseline), self.assertRaisesRegex(RuntimeError, "dedicated idle kernel"):
                _require_idle_kernel(baseline)


if __name__ == "__main__":
    unittest.main()
