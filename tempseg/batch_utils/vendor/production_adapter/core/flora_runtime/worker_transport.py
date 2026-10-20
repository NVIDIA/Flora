# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable
from flora_runtime.schema import AnnotationSource

class SubtaskAnnotationAdapterError(RuntimeError):
    pass

@runtime_checkable
class JsonPredictor(Protocol):

    def predict(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

def source_payload(source: AnnotationSource) -> dict[str, Any]:
    return {'source_id': source.source_id, 'location': str(source.location), 'duration_s': source.duration_s, 'fps': source.fps, 'metadata': dict(source.metadata), 'video_streams': [{'camera': stream.camera, 'location': str(stream.location), 'fps': stream.fps, 'width': stream.width, 'height': stream.height, 'metadata': dict(stream.metadata)} for stream in source.video_streams]}
