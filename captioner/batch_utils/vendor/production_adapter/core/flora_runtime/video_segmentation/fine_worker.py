# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from bisect import bisect_left
from collections.abc import Sequence
from fractions import Fraction
SLOT_SPACING_US = 125000

class FineWorkerError(RuntimeError):
    pass

def _nearest_index(requested_us: int, times: Sequence[Fraction], *, max_selection_error_us: int=SLOT_SPACING_US // 2) -> int:
    requested = Fraction(requested_us, 1)
    position = bisect_left(times, requested)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(times)]
    if not candidates:
        raise FineWorkerError('no decoded frame is available for an in-video slot')
    selected = min(candidates, key=lambda index: (abs(times[index] - requested), times[index], index))
    if abs(times[selected] - requested) > max_selection_error_us:
        raise FineWorkerError('nearest decoded frame is farther than half a profile slot')
    return selected
if __name__ == '__main__':
    raise SystemExit(main())
