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

from typing import Any

import pytest
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.trtllm import (
    TrtllmConfig,
    TrtllmExternalGeneration,
)


class _FakeHttpClient:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def request_json(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "payload": payload,
                "timeout_s": timeout_s,
            }
        )
        return self.responses.pop(0)


def _config(initial_policy_version: int = 3) -> TrtllmConfig:
    return {
        "backend": "trtllm",
        "model_name": "Qwen/Qwen3-32B",
        "max_new_tokens": 6,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": None,
        "stop_token_ids": [99],
        "stop_strings": None,
        "_pad_token_id": 0,
        "colocated": {
            "enabled": False,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "trtllm_cfg": {
            "rollout_base_url": "http://rollout",
            "admin_base_url": "http://admin",
            "tensor_parallel_size": 1,
            "max_model_len": 8,
            "request_timeout_s": 30,
            "admin_timeout_s": 10,
            "refit_timeout_s": 120,
            "auth_token_env": "",
            "protocol_version": "nemo-rl-trtllm-v1",
            "initial_policy_version": initial_policy_version,
            "health_check_on_init": False,
        },
    }


def _rollout_response(policy_version: int = 3) -> dict[str, Any]:
    return {
        "protocol_version": "nemo-rl-trtllm-v1",
        "policy_version": policy_version,
        "policy_update_step": policy_version,
        "outputs": [
            {
                "request_id": 0,
                "generated_token_ids": [4, 5],
                "sampled_token_logprobs": [-0.1, -0.2],
                "finish_reason": "stop",
                "accepted_draft_tokens": 2,
                "draft_tokens": 3,
                "verification_rounds": 1,
                "accepted_draft_tokens_per_round": [2],
                "draft_enabled": True,
            },
            {
                "request_id": 1,
                "generated_token_ids": [6],
                "sampled_token_logprobs": [-0.3],
                "finish_reason": "length",
                "accepted_draft_tokens": 1,
                "draft_tokens": 3,
                "verification_rounds": 2,
                "accepted_draft_tokens_per_round": [1, 0],
                "draft_enabled": True,
            },
        ],
    }


def _remove_per_round_histograms(response: dict[str, Any]) -> dict[str, Any]:
    for output in response["outputs"]:
        output.pop("accepted_draft_tokens_per_round")
    return response


def test_generate_preserves_padding_logprobs_and_specdec_counters():
    generation = TrtllmExternalGeneration(_config())
    fake_client = _FakeHttpClient([_rollout_response()])
    generation._client = fake_client
    inputs = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3], [7, 8, 0]]),
            "input_lengths": torch.tensor([3, 2]),
            "stop_strings": [None, ["</answer>"]],
        }
    )

    output = generation.generate(inputs)

    torch.testing.assert_close(
        output["output_ids"],
        torch.tensor([[1, 2, 3, 4, 5], [7, 8, 6, 0, 0]]),
    )
    torch.testing.assert_close(
        output["logprobs"],
        torch.tensor([[0.0, 0.0, 0.0, -0.1, -0.2], [0.0, 0.0, -0.3, 0.0, 0.0]]),
    )
    assert output["generation_lengths"].tolist() == [2, 1]
    assert output["truncated"].tolist() == [False, True]
    assert output["specdec_accepted_draft_tokens"].tolist() == [2, 1]
    assert output["specdec_draft_tokens"].tolist() == [3, 3]
    assert output["specdec_verification_rounds"].tolist() == [1, 2]
    assert output["specdec_accepted_draft_tokens_per_round"] == [[2], [1, 0]]
    assert output["policy_version"].tolist() == [3, 3]

    request_payload = fake_client.calls[0]["payload"]
    assert request_payload["requests"][0]["max_new_tokens"] == 5
    assert request_payload["requests"][1]["max_new_tokens"] == 6
    assert request_payload["sampling"]["return_sampled_token_logprobs"] is True


def test_generate_accepts_aggregate_specdec_counters_without_histogram():
    generation = TrtllmExternalGeneration(_config())
    generation._client = _FakeHttpClient(
        [_remove_per_round_histograms(_rollout_response())]
    )
    inputs = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3], [7, 8, 0]]),
            "input_lengths": torch.tensor([3, 2]),
        }
    )

    output = generation.generate(inputs)

    assert output["specdec_accepted_draft_tokens"].tolist() == [2, 1]
    assert output["specdec_draft_tokens"].tolist() == [3, 3]
    assert output["specdec_verification_rounds"].tolist() == [1, 2]
    assert "specdec_accepted_draft_tokens_per_round" not in output


def test_generate_rejects_stale_policy_response():
    generation = TrtllmExternalGeneration(_config())
    generation._client = _FakeHttpClient([_rollout_response(policy_version=2)])
    inputs = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    with pytest.raises(RuntimeError, match="wrong policy"):
        generation.generate(inputs)


def test_collective_refit_advances_policy_version_only_after_ack(monkeypatch):
    generation = TrtllmExternalGeneration(_config(initial_policy_version=-1))
    generation._client = _FakeHttpClient(
        [
            {
                "protocol_version": "nemo-rl-trtllm-v1",
                "ok": True,
                "policy_version": 0,
                "policy_update_step": 0,
            }
        ]
    )
    monkeypatch.setattr(ray, "put", lambda value: value)

    result = generation.update_weights_from_collective()

    assert result == [True]
    assert generation.policy_version == 0
    assert generation.policy_update_step == 0
    payload = generation._client.calls[0]["payload"]
    assert payload["next_policy_version"] == 0
    assert payload["drain"] is True
    assert payload["reset_prefix_cache"] is True
