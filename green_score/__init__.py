import os
import subprocess
import sys
import tempfile

import pandas as pd
import torch

from .green import GREEN


def _resolve_visible_gpu_ids(num_gpus):
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        device_ids = [
            device.strip() for device in visible_devices.split(",") if device.strip()
        ]
    else:
        device_ids = [str(idx) for idx in range(torch.cuda.device_count())]

    if not device_ids:
        raise ValueError("compute_green requires at least one visible GPU for vLLM.")

    if num_gpus is None:
        return device_ids

    if num_gpus < 1:
        raise ValueError("num_gpus must be at least 1.")

    if num_gpus > len(device_ids):
        raise ValueError(
            f"Requested num_gpus={num_gpus}, but only {len(device_ids)} visible GPU(s) are available."
        )

    return device_ids[:num_gpus]


def _run_green_vllm_subprocess(command, env):
    return subprocess.Popen(command, env=env)


def compute_green(
    refs,
    hyps,
    model_name="StanfordAIMI/GREEN-RadLlama2-7b",
    num_gpus=None,
):
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have the same length.")
    if len(refs) == 0:
        return [], pd.Series(dtype=object)

    if not torch.cuda.is_available():
        raise ValueError("compute_green vLLM backend requires CUDA.")

    device_ids = _resolve_visible_gpu_ids(num_gpus)
    input_df = pd.DataFrame(
        {
            "example_id": list(range(len(refs))),
            "reference": refs,
            "hypothesis": hyps,
        }
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        shard_dir = os.path.join(temp_dir, "shards")
        os.makedirs(shard_dir, exist_ok=True)

        shard_output_paths = []
        processes = []

        for shard_idx, device_id in enumerate(device_ids):
            shard_df = input_df.iloc[shard_idx :: len(device_ids)].copy()
            if shard_df.empty:
                continue

            input_path = os.path.join(shard_dir, f"input_{shard_idx}.csv")
            output_path = os.path.join(shard_dir, f"output_{shard_idx}.csv")
            shard_df.to_csv(input_path, index=False)

            command = [
                sys.executable,
                "-m",
                "green_score.green_vllm",
                "--input_file",
                input_path,
                "--output_file",
                output_path,
                "--model_name",
                model_name,
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device_id

            shard_output_paths.append(output_path)
            processes.append(_run_green_vllm_subprocess(command, env))

        failed = False
        failed_codes = []
        for process in processes:
            return_code = process.wait()
            if return_code != 0:
                failed = True
                failed_codes.append(return_code)

        if failed:
            raise subprocess.CalledProcessError(
                returncode=failed_codes[0],
                cmd="green_score.green_vllm",
            )

        shard_outputs = [pd.read_csv(path) for path in shard_output_paths]
        output_df = pd.concat(shard_outputs, ignore_index=True)

        if len(output_df) != len(input_df):
            raise AssertionError(
                f"Merged GREEN output has {len(output_df)} rows, expected {len(input_df)}."
            )

        if not output_df["example_id"].is_unique:
            duplicate_ids = output_df["example_id"][
                output_df["example_id"].duplicated()
            ].tolist()
            raise AssertionError(f"Duplicate example_id values found: {duplicate_ids[:10]}")

        input_ids = set(input_df["example_id"].tolist())
        output_ids = set(output_df["example_id"].tolist())
        if input_ids != output_ids:
            missing_ids = sorted(input_ids - output_ids)
            extra_ids = sorted(output_ids - input_ids)
            raise AssertionError(
                f"GREEN output example_id mismatch. Missing: {missing_ids[:10]}; Extra: {extra_ids[:10]}"
            )

        output_df = output_df.sort_values("example_id").reset_index(drop=True)
        return output_df["green_score"].tolist(), output_df["green_analysis"]
