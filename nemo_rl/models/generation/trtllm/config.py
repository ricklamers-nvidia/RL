# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TypedDict

from nemo_rl.models.generation.interfaces import GenerationConfig


class TrtllmSpecificArgs(TypedDict):
    """Configuration for a persistent external TRT-LLM target."""

    rollout_base_url: str
    admin_base_url: str
    tensor_parallel_size: int
    max_model_len: int
    request_timeout_s: float
    admin_timeout_s: float
    refit_timeout_s: float
    auth_token_env: str
    protocol_version: str
    initial_policy_version: int
    health_check_on_init: bool


class TrtllmConfig(GenerationConfig):
    trtllm_cfg: TrtllmSpecificArgs
