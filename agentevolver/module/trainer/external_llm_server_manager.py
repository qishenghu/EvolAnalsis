"""External vLLM rollout server manager.

Replaces BaAsyncLLMServerManager when ``actor_rollout_ref.rollout.name ==
"external_vllm"``: instead of verl's colocated AsyncvLLMServer ray actors, the
rollout runs on standalone ``vllm serve`` processes (started via
``start_rollout_servers.sh``) whose addresses come from
``actor_rollout_ref.rollout.external_server_addresses``.

Exposes the exact surface ParallelEnvManager consumes:
  - submit_chat_completions(messages, sampling_params)  (sync blocking)
  - chat(messages, sampling_params)
  - chat_scheduler.model_name
  - chat_scheduler.completion_callback.tokenizer
  - chat_scheduler.weighted_addresses
plus wake_up()/sleep() so the trainer's existing call sites work unchanged.

Weight sync: wake_up() first restores any sleeping server and then lazily syncs
the FSDP actor weights whenever notify_weights_updated() was called since the
last sync (and once at startup). Mechanism:
worker_group.save_rollout_weights() exports bf16
safetensors, then each server gets
POST /collective_rpc {"method": "reload_weights_from_disk", "args": [dir]}
(dev-mode endpoint, VLLM_SERVER_DEV_MODE=1, worker extension
duet_vllm_worker_ext.RolloutWeightReloadExtension), followed by
/reset_prefix_cache.
"""

import asyncio
import hashlib
import json
import os
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional

import requests
from loguru import logger
from omegaconf import DictConfig

from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler


class ExternalLLMServerManager:
    """Manage a static pool of external OpenAI-compatible vLLM servers."""

    def __init__(self, config: DictConfig, worker_group=None):
        self.full_config = config
        self.config = config.actor_rollout_ref
        self.worker_group = worker_group

        addresses = list(self.config.rollout.get("external_server_addresses", None) or [])
        assert addresses, "actor_rollout_ref.rollout.external_server_addresses must be a non-empty list for rollout.name=external_vllm"
        self.server_addresses = addresses
        self._admin_timeout = float(self.config.rollout.get("external_admin_timeout", 1800))
        self._request_timeout = float(
            self.config.rollout.get("external_request_timeout", 600)
        )
        self._sleep_between_steps = bool(self.config.rollout.get("external_sleep_between_steps", False))
        self._sync_enabled = bool(self.config.rollout.get("external_weight_sync", True))
        sync_dir = self.config.rollout.get("external_sync_dir", None)
        if not sync_dir:
            run_name = str(self.full_config.trainer.get("experiment_name", "default"))
            # tmpfs by default: the sync writes the full bf16 actor every step and
            # each server reads it back, so disk I/O dominated the step time (88s
            # for 4B). Fall back to the working tree when /dev/shm is too small.
            shm = "/dev/shm"
            need_gb = 40
            try:
                st = os.statvfs(shm)
                shm_free_gb = st.f_bavail * st.f_frsize / 1e9
            except OSError:
                shm_free_gb = 0
            if shm_free_gb >= need_gb:
                sync_dir = os.path.join(shm, "duet_rollout_sync", run_name)
            else:
                sync_dir = os.path.join(os.getcwd(), "tmp_rollout_sync", run_name)
        self._sync_dir = sync_dir

        # weights on the servers start from the on-disk base model; force one sync
        # so training resumes / first rollout use the FSDP weights.
        self._weights_dirty = self._sync_enabled
        self._sync_metrics: Dict[str, float] = {}
        self._unsupported_admin_routes = set()
        # The servers outlive this manager and may have been left in level-1
        # sleep by a previous trainer.  /health remains successful in that
        # state, so the initial state cannot be inferred from the health check.
        # Keep it unknown until the first idempotent /wake_up succeeds; this is
        # especially important because reloading weights while CUDA storage is
        # released can corrupt the next wake-up.
        self._is_sleeping: Optional[bool] = None

        self._check_servers_alive()

        # Init chat scheduler in a separate thread (mirrors verl AsyncLLMServerManager).
        self.chat_scheduler: ChatCompletionScheduler = None
        self.chat_scheduler_exception: Exception = None
        self.chat_scheduler_loop = None
        self.chat_scheduler_ready = threading.Event()
        self.chat_scheduler_thread = threading.Thread(target=self._init_chat_scheduler, daemon=True)
        self.chat_scheduler_thread.start()
        self.chat_scheduler_ready.wait()
        if self.chat_scheduler_exception is not None:
            raise self.chat_scheduler_exception

        logger.info(f"ExternalLLMServerManager: {len(addresses)} servers {addresses}, model_name={self.chat_scheduler.model_name}, sync_dir={self._sync_dir}")

    def _init_chat_scheduler(self):
        self.chat_scheduler_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.chat_scheduler_loop)
        try:
            self.chat_scheduler = ChatCompletionScheduler(
                config=self.full_config,
                server_addresses=self.server_addresses,
            )
        except Exception as e:
            logger.exception(f"chat_scheduler init error: {e}")
            self.chat_scheduler_exception = e
        finally:
            self.chat_scheduler_ready.set()
        self.chat_scheduler_loop.run_forever()

    # ------------------------------------------------------------------
    # Generation surface (same as BaAsyncLLMServerManager)
    # ------------------------------------------------------------------
    def chat(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> str:
        self.submit_chat_completions(messages.copy(), sampling_params)
        return messages[-1]["content"]

    def submit_chat_completions(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Dict[str, Any],
        request_id: Optional[str] = None,
    ):
        """Submit a chat completion request to chat scheduler and wait until it is done."""
        assert self.chat_scheduler is not None, "chat scheduler is not initialized."
        future = asyncio.run_coroutine_threadsafe(
            self.chat_scheduler._submit_chat_completions_semaphore(
                messages=messages,
                request_id=request_id,
                sampling_params=sampling_params,
            ),
            self.chat_scheduler_loop,
        )
        try:
            future.result(timeout=self._request_timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise RuntimeError(
                f"external rollout request exceeded {self._request_timeout}s"
            ) from exc

    # ------------------------------------------------------------------
    # Admin (dev-mode HTTP endpoints; no-op with warning if unsupported)
    # ------------------------------------------------------------------
    def _admin_post(self, address: str, route: str, timeout: Optional[float] = None, **params) -> bool:
        if route in self._unsupported_admin_routes:
            return False
        url = f"http://{address}{route}"
        try:
            resp = requests.post(url, params=params or None, timeout=timeout or self._admin_timeout)
        except requests.RequestException as e:
            logger.warning(f"admin POST {url} failed: {e}")
            return False
        if resp.status_code == 404:
            logger.warning(f"admin route {route} unsupported on {address} (404) — is VLLM_SERVER_DEV_MODE=1 set? Disabling this route.")
            self._unsupported_admin_routes.add(route)
            return False
        if resp.status_code != 200:
            logger.warning(f"admin POST {url} -> {resp.status_code}: {resp.text[:500]}")
            return False
        return True

    def _check_servers_alive(self):
        for address in self.server_addresses:
            try:
                resp = requests.get(f"http://{address}/health", timeout=30)
                assert resp.status_code == 200
            except Exception as e:
                raise RuntimeError(f"external rollout server {address} is not healthy (start it with start_rollout_servers.sh): {e}") from e

    def wake_up(self):
        """Wake sleeping servers first, then sync dirty actor weights."""
        # A level-1 sleeping vLLM worker has released its CUDA model storage.
        # Wake it before an in-place reload, then sync the newest actor weights.
        # A newly attached manager does not own the servers' prior state.  vLLM
        # treats /wake_up on an already-awake executor as an idempotent no-op,
        # so reconcile the unknown initial state even when this run does not
        # plan to sleep between steps.
        if self._is_sleeping is not False:
            for address in self.server_addresses:
                if not self._admin_post(address, "/wake_up"):
                    raise RuntimeError(f"failed to wake rollout server {address}")
            self._is_sleeping = False
            self._check_servers_alive()
        synced_for_generation = False
        if self._weights_dirty and self._sync_enabled:
            self.sync_rollout_weights()
            self._weights_dirty = False
            synced_for_generation = True
        self._check_servers_alive()
        return synced_for_generation

    def sleep(self):
        """Optionally put servers to sleep (dedicated-GPU servers usually stay awake)."""
        if self._sleep_between_steps and not self._is_sleeping:
            for address in self.server_addresses:
                if not self._admin_post(address, "/sleep", level=1):
                    raise RuntimeError(f"failed to sleep rollout server {address}")
            self._is_sleeping = True

    def reset_prefix_cache(self):
        for address in self.server_addresses:
            if not self._admin_post(address, "/reset_prefix_cache"):
                raise RuntimeError(
                    f"failed to reset prefix cache on rollout server {address}"
                )

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------
    def notify_weights_updated(self):
        """Trainer calls this after update_actor; next wake_up() re-syncs."""
        self._weights_dirty = True

    def get_sync_metrics(self) -> Dict[str, float]:
        return {f"external_rollout/{k}": v for k, v in self._sync_metrics.items()}

    def _collective_rpc(self, address: str, method: str, args: list) -> list:
        """Call one server's workers and require a structured result payload."""
        url = f"http://{address}/collective_rpc"
        try:
            resp = requests.post(
                url,
                json={"method": method, "args": args},
                timeout=self._admin_timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"collective RPC {method} failed on {address}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise RuntimeError(
                f"collective RPC {method} failed on {address} "
                f"({resp.status_code}): {resp.text[:500]}"
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"collective RPC {method} on {address} returned non-JSON: "
                f"{resp.text[:500]}"
            ) from exc
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or not results:
            raise RuntimeError(
                f"collective RPC {method} on {address} returned no worker results: "
                f"{payload!r}"
            )
        return results

    def sync_rollout_weights(self):
        """Export FSDP actor weights to disk and hot-reload them into every server."""
        assert self.worker_group is not None, "worker_group is required for weight sync"
        t0 = time.time()
        os.makedirs(self._sync_dir, exist_ok=True)
        self.worker_group.save_rollout_weights(self._sync_dir)
        t_export = time.time() - t0

        t1 = time.time()
        reload_fingerprint = None
        reload_stats = None
        for address in self.server_addresses:
            results = self._collective_rpc(
                address,
                "reload_weights_from_disk",
                [self._sync_dir],
            )
            if not all(isinstance(result, dict) for result in results):
                raise RuntimeError(
                    f"weight reload on {address} returned malformed results: {results!r}"
                )
            for result in results:
                if (
                    int(result.get("num_files", 0)) <= 0
                    or int(result.get("num_exported_tensors", 0)) <= 0
                    or int(result.get("num_loaded_params", 0)) <= 0
                ):
                    raise RuntimeError(
                        f"weight reload on {address} reported incomplete load: {result!r}"
                    )
            current_fingerprint = json.dumps(results, sort_keys=True)
            if reload_fingerprint is None:
                reload_fingerprint = current_fingerprint
                reload_stats = results[0]
            elif current_fingerprint != reload_fingerprint:
                raise RuntimeError(
                    "rollout servers reported different weight reload contracts: "
                    f"first={reload_fingerprint}, {address}={current_fingerprint}"
                )

        checksum_patterns = list(
            self.config.rollout.get(
                "external_weight_checksum_patterns",
                ["embed_tokens", "layers.0.", "layers.31.", "lm_head"],
            )
        )
        checksum_fingerprint = None
        checksum_count = 0
        for address in self.server_addresses:
            results = self._collective_rpc(
                address,
                "param_checksums",
                [checksum_patterns],
            )
            current_fingerprint = hashlib.sha256(
                json.dumps(results, sort_keys=True).encode("utf-8")
            ).hexdigest()
            current_count = sum(
                len(result) for result in results if isinstance(result, dict)
            )
            if current_count <= 0:
                raise RuntimeError(
                    f"parameter checksum probe matched no tensors on {address}: "
                    f"patterns={checksum_patterns!r}"
                )
            if checksum_fingerprint is None:
                checksum_fingerprint = current_fingerprint
                checksum_count = current_count
            elif current_fingerprint != checksum_fingerprint:
                raise RuntimeError(
                    "rollout server parameter checksums differ after reload: "
                    f"expected={checksum_fingerprint}, "
                    f"{address}={current_fingerprint}"
                )
        self.reset_prefix_cache()
        self._check_servers_alive()
        t_reload = time.time() - t1

        self._sync_metrics = {
            "weight_sync_export_s": t_export,
            "weight_sync_reload_s": t_reload,
            "weight_sync_total_s": time.time() - t0,
            "weight_sync_exported_tensors": float(
                reload_stats.get("num_exported_tensors", 0)
            ),
            "weight_sync_loaded_params": float(
                reload_stats.get("num_loaded_params", 0)
            ),
            "weight_sync_checksum_params": float(checksum_count),
        }
        logger.info(
            "external rollout weight sync verified: "
            f"export={t_export:.1f}s reload={t_reload:.1f}s "
            f"exported={reload_stats.get('num_exported_tensors')} "
            f"loaded={reload_stats.get('num_loaded_params')} "
            f"checksum_params={checksum_count} "
            f"checksum={checksum_fingerprint[:16]} "
            f"({len(self.server_addresses)} servers)"
        )
