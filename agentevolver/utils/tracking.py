# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modifications copyright 2025 Alibaba Tongyi EconML Lab. and/or its affiliates
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
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import os
from contextlib import nullcontext
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Union


REQUIRED_WANDB_PROJECT = "agentevolver"
REQUIRED_EXPERIMENT_LOGGERS = ("console", "wandb")
WANDB_RUNTIME_ENV_KEYS = (
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
    "WANDB_MODE",
    "WANDB_ENTITY",
    "WANDB_CONSOLE",
    "WANDB_IDENTITY_TOKEN_FILE",
    "WANDB_CREDENTIALS_FILE",
)
WANDB_SECRET_ENV_KEYS = (
    "WANDB_API_KEY",
    "WANDB_IDENTITY_TOKEN_FILE",
    "WANDB_CREDENTIALS_FILE",
)


def normalize_experiment_loggers(default_backend: Union[str, List[str], None]) -> List[str]:
    """Return logger backends with both console and online W&B required."""

    if isinstance(default_backend, str):
        normalized = [default_backend]
    elif default_backend is None:
        normalized = []
    else:
        normalized = list(default_backend)

    for backend in REQUIRED_EXPERIMENT_LOGGERS:
        if backend not in normalized:
            normalized.append(backend)
    return normalized


def enforce_online_wandb_config(config):
    """Normalize the training config to the repository's mandatory W&B policy.

    This mutates and returns ``config``.  It intentionally supports both plain
    dictionaries and OmegaConf ``DictConfig`` objects so the same contract is
    applied by ``launcher.py`` and by direct ``python -m`` entry points.
    """

    if config is None:
        raise RuntimeError("Training config is required for mandatory W&B tracking.")

    def writable(node):
        try:
            from omegaconf import OmegaConf, open_dict

            if OmegaConf.is_config(node):
                return open_dict(node)
        except ImportError:
            pass
        return nullcontext()

    with writable(config):
        trainer = config.get("trainer")
        if trainer is None:
            try:
                config["trainer"] = {}
                trainer = config["trainer"]
            except Exception:
                raise RuntimeError(
                    "Training config must contain a mutable 'trainer' section for "
                    "mandatory W&B tracking."
                ) from None

        with writable(trainer):
            experiment_name = trainer.get("experiment_name")
            if experiment_name is None or not str(experiment_name).strip():
                raise RuntimeError(
                    "trainer.experiment_name must be set; it is the mandatory "
                    "W&B run name."
                )

            trainer["logger"] = normalize_experiment_loggers(
                trainer.get("logger")
            )
            trainer["project_name"] = REQUIRED_WANDB_PROJECT
    return config


def require_wandb_online_environment(
    env: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Reject no-upload modes and make the W&B SDK's online mode explicit."""

    if env is None:
        env = os.environ

    if str(env.get("WANDB_API_KEY", "")).strip():
        raise RuntimeError(
            "Plaintext WANDB_API_KEY environment authentication is not allowed "
            "for Ray training because it can be inherited or serialized. Unset "
            "WANDB_API_KEY and persist credentials first with "
            "`env -u WANDB_API_KEY wandb login --verify` (the credential file "
            "must be mode 0600 and accessible only to the current user)."
        )

    configured_mode = env.get("WANDB_MODE")
    if configured_mode is not None:
        normalized_mode = str(configured_mode).strip().lower()
        if normalized_mode != "online":
            raise RuntimeError(
                "Online W&B is mandatory: WANDB_MODE="
                f"{configured_mode!r} is not allowed. Unset it or set "
                "WANDB_MODE=online."
            )

    disabled = str(env.get("WANDB_DISABLED", "")).strip().lower()
    if disabled not in ("", "0", "false", "no", "off"):
        raise RuntimeError(
            "Online W&B is mandatory: WANDB_DISABLED must not disable uploads."
        )

    # An explicit value prevents a worker-local W&B settings file from
    # silently selecting offline/dryrun mode after the driver preflight.
    env["WANDB_MODE"] = "online"
    return env


def get_wandb_runtime_env(
    env: MutableMapping[str, str] | None = None,
) -> Dict[str, str]:
    """Return the minimal W&B environment needed by the Ray TaskRunner."""

    if env is None:
        env = os.environ
    require_wandb_online_environment(env)
    return {
        key: str(env[key])
        for key in WANDB_RUNTIME_ENV_KEYS
        if env.get(key) is not None and str(env.get(key)) != ""
    }


def preflight_wandb_online(
    *,
    project_name: str,
    experiment_name: str,
    timeout: int = 20,
    https_proxy: str | None = None,
) -> None:
    """Verify W&B network and authentication without creating a run.

    The exception deliberately omits the underlying SDK message because some
    HTTP/client failures can include credentials or signed request details.
    """

    if project_name != REQUIRED_WANDB_PROJECT:
        raise RuntimeError(
            f"W&B project must be {REQUIRED_WANDB_PROJECT!r}, got "
            f"{project_name!r}."
        )
    if experiment_name is None or not str(experiment_name).strip():
        raise RuntimeError("A non-empty W&B run name is required.")

    require_wandb_online_environment()
    proxy_sentinel = object()
    previous_proxy_values = {}
    if https_proxy:
        for proxy_key in ("HTTPS_PROXY", "https_proxy"):
            previous_proxy_values[proxy_key] = os.environ.get(
                proxy_key, proxy_sentinel
            )
            os.environ[proxy_key] = str(https_proxy)
    try:
        import wandb

        # Bound any accidental interactive prompt to one second.  The login
        # call verifies an API key when present; ``Api.viewer`` additionally
        # validates identity-token auth and makes a read-only network request.
        wandb.login(force=True, timeout=1, verify=True)
        api = wandb.Api(timeout=timeout)
        _ = api.viewer
    except Exception as exc:
        error_type = type(exc).__name__
        raise RuntimeError(
            "W&B online/auth preflight failed before Ray/GPU initialization "
            f"({error_type}). Verify network access and run "
            "`wandb login --verify`; no training workers were started."
        ) from None
    finally:
        for proxy_key, previous_value in previous_proxy_values.items():
            if previous_value is proxy_sentinel:
                os.environ.pop(proxy_key, None)
            else:
                os.environ[proxy_key] = previous_value


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = ["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console", "clearml"]

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = "console", config=None):
        default_backend = normalize_experiment_loggers(default_backend)
        require_wandb_online_environment()
        if experiment_name is None or not str(experiment_name).strip():
            raise RuntimeError("A non-empty W&B run name is required.")
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}
        self._finished = False

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            # Metrics and tables are still uploaded, while avoiding a second
            # copy of the very large trainer stdout in the local W&B cache.
            settings_kwargs = {"console": "off"}
            if config and config["trainer"].get("wandb_proxy", None):
                settings_kwargs["https_proxy"] = config["trainer"]["wandb_proxy"]
            settings = wandb.Settings(**settings_kwargs)
            wandb.init(
                project=REQUIRED_WANDB_PROJECT,
                name=experiment_name,
                config=config,
                settings=settings,
                mode="online",
            )
            self.logger["wandb"] = wandb

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
                settings=swanlab.Settings(backup=False, log_proxy_type='none')
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter()

        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def finish(self, exit_code: int = 0):
        """Finish every backend once, preserving W&B's success/failure state."""

        if getattr(self, "_finished", False):
            return
        # Mark first so a partially failing SDK finalizer is never retried by
        # ``__del__`` with an incorrect success exit code.
        self._finished = True
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=exit_code)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=exit_code)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()

        if "clearnml" in self.logger:
            self.logger["clearnml"].finish()

    def __del__(self):
        try:
            # Canonical training paths finish explicitly.  Reaching the
            # destructor first is an unclean/unknown exit and must never be
            # reported to W&B as a successful run.
            self.finish(exit_code=1)
        except Exception:
            # Destructors must not obscure an active training exception.  The
            # explicit TaskRunner success/failure paths call ``finish`` first.
            pass


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, (int, float, np.floating, np.integer)):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This invocation of ClearML logger\'s function is incorrect so this attribute was dropped. ')

    def finish(self):
        self._task.mark_completed()


class _TensorboardAdapter:
    def __init__(self):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow

        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

    def log_generations_to_wandb(self, samples, step):
        """Log samples to wandb as a table"""
        import wandb

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], [])

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = f"""
            input: {sample[0]}
            
            ---
            
            output: {sample[1]}
            
            ---
            
            score: {sample[2]}
            """
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_text_list}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()
