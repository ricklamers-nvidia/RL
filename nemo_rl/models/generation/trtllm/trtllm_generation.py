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

"""External TRT-LLM generation client for persistent LPU-backed targets."""

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.trtllm.config import TrtllmConfig


class _JsonHttpClient:
    def __init__(self, auth_token_env: str):
        self._auth_token_env = auth_token_env

    def request_json(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"

        if self._auth_token_env:
            token = os.environ.get(self._auth_token_env)
            if not token:
                raise RuntimeError(
                    f"Required TRT-LLM auth token environment variable "
                    f"{self._auth_token_env!r} is not set"
                )
            headers["Authorization"] = f"Bearer {token}"

        request = Request(url=url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout_s) as response:
                response_body = response.read()
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"TRT-LLM request {method} {url} failed with HTTP {exc.code}: {body}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"TRT-LLM request {method} {url} failed: {exc.reason}"
            ) from exc

        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"TRT-LLM request {method} {url} returned invalid JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(
                f"TRT-LLM request {method} {url} returned "
                f"{type(decoded).__name__}, expected an object"
            )
        return decoded


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _serialize_state_dict_info(
    state_dict_info: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    serialized: dict[str, dict[str, Any]] = {}
    for name, value in state_dict_info.items():
        shape, dtype = value
        dtype_name = str(dtype)
        if dtype_name.startswith("torch."):
            dtype_name = dtype_name.removeprefix("torch.")
        serialized[name] = {
            "shape": list(shape),
            "dtype": dtype_name,
        }
    return serialized


class TrtllmExternalGeneration(GenerationInterface):
    """Use a separately deployed TRT-LLM/LPU service for generation and refits."""

    def __init__(self, config: TrtllmConfig):
        self.cfg = config
        self._trtllm_cfg = config["trtllm_cfg"]
        self._protocol_version = self._trtllm_cfg["protocol_version"]
        self._policy_version = self._trtllm_cfg["initial_policy_version"]
        self._policy_update_step = self._policy_version
        self._client = _JsonHttpClient(self._trtllm_cfg["auth_token_env"])

        self._validate_config()
        if self._trtllm_cfg["health_check_on_init"]:
            self._validate_health()

    @property
    def policy_version(self) -> int:
        return self._policy_version

    @property
    def policy_update_step(self) -> int:
        return self._policy_update_step

    def _validate_config(self) -> None:
        required_keys = [
            key for key in TrtllmConfig.__required_keys__ if key not in self.cfg
        ]
        if required_keys:
            raise ValueError(
                f"TRT-LLM configuration is missing required keys: {required_keys}"
            )
        if self._trtllm_cfg["tensor_parallel_size"] < 1:
            raise ValueError("trtllm_cfg.tensor_parallel_size must be >= 1")
        if self._trtllm_cfg["max_model_len"] < 1:
            raise ValueError("trtllm_cfg.max_model_len must be >= 1")
        if self.cfg["max_new_tokens"] < 1:
            raise ValueError("generation.max_new_tokens must be >= 1")
        if not self._protocol_version:
            raise ValueError("trtllm_cfg.protocol_version must not be empty")

    def _rollout_request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.request_json(
            method,
            _join_url(self._trtllm_cfg["rollout_base_url"], path),
            payload=payload,
            timeout_s=self._trtllm_cfg["request_timeout_s"],
        )

    def _admin_request(
        self,
        path: str,
        *,
        payload: dict[str, Any],
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return self._client.request_json(
            "POST",
            _join_url(self._trtllm_cfg["admin_base_url"], path),
            payload=payload,
            timeout_s=timeout_s or self._trtllm_cfg["admin_timeout_s"],
        )

    def _validate_protocol_response(self, response: dict[str, Any]) -> None:
        actual = response.get("protocol_version")
        if actual != self._protocol_version:
            raise RuntimeError(
                f"TRT-LLM protocol mismatch: expected {self._protocol_version!r}, "
                f"got {actual!r}"
            )

    def _validate_health(self) -> None:
        health = self._rollout_request("GET", "/healthz")
        self._validate_protocol_response(health)
        if health.get("status") != "ready":
            raise RuntimeError(f"TRT-LLM rollout service is not ready: {health}")

    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
    ) -> list[ray.ObjectRef]:
        inference_world_size = self._trtllm_cfg["tensor_parallel_size"]
        if world_size != train_world_size + inference_world_size:
            raise ValueError(
                f"Collective world size mismatch: world_size={world_size}, "
                f"train_world_size={train_world_size}, "
                f"inference_world_size={inference_world_size}"
            )
        response = self._admin_request(
            "/v1/admin/collective/init",
            payload={
                "protocol_version": self._protocol_version,
                "master_address": ip,
                "port": port,
                "world_size": world_size,
                "train_world_size": train_world_size,
                "inference_world_size": inference_world_size,
            },
        )
        self._validate_protocol_response(response)
        return [ray.put(response.get("ok") is True)]

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        serialized = _serialize_state_dict_info(state_dict_info)
        response = self._admin_request(
            "/v1/admin/refit/metadata",
            payload={
                "protocol_version": self._protocol_version,
                "parameters": serialized,
            },
        )
        self._validate_protocol_response(response)
        if response.get("parameter_count") != len(serialized):
            raise RuntimeError(
                "TRT-LLM target did not acknowledge the complete refit metadata: "
                f"expected {len(serialized)}, got {response.get('parameter_count')}"
            )

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        next_policy_version = self._policy_version + 1
        response = self._admin_request(
            "/v1/admin/refit/collective",
            payload={
                "protocol_version": self._protocol_version,
                "next_policy_version": next_policy_version,
                "next_policy_update_step": next_policy_version,
                "drain": True,
                "reset_prefix_cache": True,
                "reopen_admission": True,
            },
            timeout_s=self._trtllm_cfg["refit_timeout_s"],
        )
        self._validate_protocol_response(response)
        success = response.get("ok") is True
        if success:
            applied_version = response.get("policy_version")
            applied_step = response.get("policy_update_step")
            if (
                applied_version != next_policy_version
                or applied_step != next_policy_version
            ):
                raise RuntimeError(
                    "TRT-LLM target acknowledged an unexpected policy version: "
                    f"version={applied_version}, update_step={applied_step}, "
                    f"expected={next_policy_version}"
                )
            self._policy_version = next_policy_version
            self._policy_update_step = next_policy_version
        return [ray.put(success)]

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        raise NotImplementedError(
            "The persistent TRT-LLM target supports non-colocated collective refits only"
        )

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        if self._policy_version < 0:
            raise RuntimeError(
                "TRT-LLM target has not received its initial policy refit"
            )
        response = self._admin_request(
            "/v1/admin/admission/open",
            payload={
                "protocol_version": self._protocol_version,
                "policy_version": self._policy_version,
                "policy_update_step": self._policy_update_step,
            },
        )
        self._validate_protocol_response(response)
        return response.get("ok") is True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        response = self._admin_request(
            "/v1/admin/admission/close",
            payload={
                "protocol_version": self._protocol_version,
                "drain": True,
            },
        )
        self._validate_protocol_response(response)
        return response.get("ok") is True

    def invalidate_kv_cache(self) -> bool:
        response = self._admin_request(
            "/v1/admin/cache/reset",
            payload={"protocol_version": self._protocol_version},
        )
        self._validate_protocol_response(response)
        return response.get("ok") is True

    def generate(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        if self._policy_version < 0:
            raise RuntimeError(
                "TRT-LLM target has not received its initial policy refit"
            )
        verify_right_padding(data, pad_value=self.cfg["_pad_token_id"])

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        stop_strings = data.get("stop_strings", [None] * len(input_lengths))
        requests = []
        for index, input_length_tensor in enumerate(input_lengths):
            input_length = int(input_length_tensor.item())
            max_new_tokens = min(
                self.cfg["max_new_tokens"],
                self._trtllm_cfg["max_model_len"] - input_length,
            )
            if max_new_tokens < 1:
                raise ValueError(
                    f"Prompt at batch index {index} has length {input_length}, "
                    f"leaving no room in max_model_len="
                    f"{self._trtllm_cfg['max_model_len']}"
                )
            requests.append(
                {
                    "request_id": index,
                    "input_ids": input_ids[index, :input_length].tolist(),
                    "max_new_tokens": max_new_tokens,
                    "stop_strings": stop_strings[index],
                }
            )

        payload = {
            "protocol_version": self._protocol_version,
            "policy_version": self._policy_version,
            "policy_update_step": self._policy_update_step,
            "sampling": {
                "temperature": 0.0 if greedy else self.cfg["temperature"],
                "top_p": 1.0 if greedy else self.cfg["top_p"],
                "top_k": 1 if greedy else self.cfg["top_k"],
                "stop_token_ids": self.cfg["stop_token_ids"],
                "return_sampled_token_logprobs": True,
            },
            "requests": requests,
        }
        response = self._rollout_request(
            "POST", "/v1/rollouts/generate", payload=payload
        )
        self._validate_protocol_response(response)
        self._validate_response_version(response)
        return self._build_generation_output(input_ids, input_lengths, response)

    def _validate_response_version(self, response: dict[str, Any]) -> None:
        response_version = response.get("policy_version")
        response_step = response.get("policy_update_step")
        if (
            response_version != self._policy_version
            or response_step != self._policy_update_step
        ):
            raise RuntimeError(
                "TRT-LLM returned rollout data from the wrong policy: "
                f"version={response_version}, update_step={response_step}, "
                f"expected version={self._policy_version}, "
                f"update_step={self._policy_update_step}"
            )

    def _build_generation_output(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        response: dict[str, Any],
    ) -> BatchedDataDict[GenerationOutputSpec]:
        outputs = response.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != len(input_lengths):
            raise RuntimeError(
                f"TRT-LLM returned {len(outputs) if isinstance(outputs, list) else 'invalid'} "
                f"outputs for a batch of {len(input_lengths)}"
            )
        outputs_by_id: dict[int, dict[str, Any]] = {}
        for output in outputs:
            if not isinstance(output, dict):
                raise RuntimeError("TRT-LLM returned a non-object rollout output")
            request_id = output.get("request_id")
            if (
                not isinstance(request_id, int)
                or request_id < 0
                or request_id >= len(input_lengths)
                or request_id in outputs_by_id
            ):
                raise RuntimeError(
                    f"TRT-LLM returned invalid or duplicate request_id={request_id!r}"
                )
            outputs_by_id[request_id] = output
        ordered_outputs = [outputs_by_id[index] for index in range(len(input_lengths))]

        rows: list[list[int]] = []
        logprob_rows: list[list[float]] = []
        generation_lengths: list[int] = []
        unpadded_lengths: list[int] = []
        truncated: list[bool] = []
        accepted_totals: list[int] = []
        draft_totals: list[int] = []
        verification_rounds: list[int] = []
        accepted_per_round: list[list[int]] = []
        per_round_available: bool | None = None
        finish_reasons: list[str] = []
        draft_enabled: list[bool] = []

        for index, output in enumerate(ordered_outputs):
            generated_ids = output.get("generated_token_ids")
            sampled_logprobs = output.get("sampled_token_logprobs")
            if not isinstance(generated_ids, list) or not all(
                isinstance(token_id, int) for token_id in generated_ids
            ):
                raise RuntimeError(
                    f"TRT-LLM output {index} has invalid generated_token_ids"
                )
            if not isinstance(sampled_logprobs, list) or len(sampled_logprobs) != len(
                generated_ids
            ):
                raise RuntimeError(
                    f"TRT-LLM output {index} has {len(sampled_logprobs) if isinstance(sampled_logprobs, list) else 'invalid'} "
                    f"logprobs for {len(generated_ids)} generated tokens"
                )

            accepted = int(output.get("accepted_draft_tokens", 0))
            drafted = int(output.get("draft_tokens", 0))
            rounds = int(output.get("verification_rounds", 0))
            per_round = output.get("accepted_draft_tokens_per_round")
            has_per_round = per_round is not None
            if per_round_available is None:
                per_round_available = has_per_round
            elif per_round_available != has_per_round:
                raise RuntimeError(
                    "TRT-LLM returned accepted-draft histograms for only part "
                    "of the batch"
                )
            if accepted < 0 or drafted < 0 or rounds < 0 or accepted > drafted:
                raise RuntimeError(
                    f"TRT-LLM output {index} has inconsistent speculative-decoding counters"
                )
            if has_per_round and (
                not isinstance(per_round, list)
                or not all(isinstance(value, int) and value >= 0 for value in per_round)
                or len(per_round) != rounds
                or sum(per_round) != accepted
            ):
                raise RuntimeError(
                    f"TRT-LLM output {index} has an inconsistent "
                    "accepted-draft histogram"
                )

            input_length = int(input_lengths[index].item())
            prompt_ids = input_ids[index, :input_length].tolist()
            rows.append(prompt_ids + generated_ids)
            logprob_rows.append([0.0] * input_length + sampled_logprobs)
            generation_lengths.append(len(generated_ids))
            unpadded_lengths.append(input_length + len(generated_ids))
            finish_reason = str(output.get("finish_reason", ""))
            finish_reasons.append(finish_reason)
            truncated.append(finish_reason == "length")
            accepted_totals.append(accepted)
            draft_totals.append(drafted)
            verification_rounds.append(rounds)
            if has_per_round:
                accepted_per_round.append(per_round)
            draft_enabled.append(bool(output.get("draft_enabled", drafted > 0)))

        max_length = max(unpadded_lengths, default=0)
        output_ids = torch.full(
            (len(rows), max_length),
            self.cfg["_pad_token_id"],
            dtype=input_ids.dtype,
        )
        logprobs = torch.zeros((len(rows), max_length), dtype=torch.float32)
        for index, (row, logprob_row) in enumerate(zip(rows, logprob_rows)):
            output_ids[index, : len(row)] = torch.tensor(row, dtype=input_ids.dtype)
            logprobs[index, : len(logprob_row)] = torch.tensor(
                logprob_row, dtype=torch.float32
            )

        output_data: dict[str, Any] = {
            "output_ids": output_ids,
            "generation_lengths": torch.tensor(generation_lengths, dtype=torch.int64),
            "unpadded_sequence_lengths": torch.tensor(
                unpadded_lengths, dtype=torch.int64
            ),
            "logprobs": logprobs,
            "truncated": torch.tensor(truncated, dtype=torch.bool),
            "specdec_accepted_draft_tokens": torch.tensor(
                accepted_totals, dtype=torch.int64
            ),
            "specdec_draft_tokens": torch.tensor(draft_totals, dtype=torch.int64),
            "specdec_verification_rounds": torch.tensor(
                verification_rounds, dtype=torch.int64
            ),
            "specdec_draft_enabled": torch.tensor(draft_enabled, dtype=torch.bool),
            "policy_version": torch.full(
                (len(rows),), self._policy_version, dtype=torch.int64
            ),
            "policy_update_step": torch.full(
                (len(rows),), self._policy_update_step, dtype=torch.int64
            ),
            "finish_reasons": finish_reasons,
        }
        if per_round_available:
            output_data["specdec_accepted_draft_tokens_per_round"] = accepted_per_round
        generation_output = BatchedDataDict[GenerationOutputSpec](output_data)
        verify_right_padding(generation_output, pad_value=self.cfg["_pad_token_id"])
        return generation_output
