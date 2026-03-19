import os
import subprocess
import tempfile

import pandas as pd
import torch

from .green import GREEN


def compute_green(
    refs,
    hyps,
    model_name="StanfordAIMI/GREEN-RadLlama2-7b",
    batch_size=8,
    num_gpus=None,
):
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have the same length.")

    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if num_gpus < 1:
        raise ValueError("num_gpus must be at least 1.")

    input_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    output_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    input_path = input_file.name
    output_path = output_file.name
    input_file.close()
    output_file.close()

    try:
        pd.DataFrame({"reference": refs, "hypothesis": hyps}).to_csv(
            input_path, index=False
        )

        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                "-m",
                "green_score.green",
                "--input_file",
                input_path,
                "--output_file",
                output_path,
                "--model_name",
                model_name,
                "--batch_size",
                str(batch_size),
            ],
            check=True,
        )

        results = pd.read_csv(output_path)
        return results["green_score"].tolist(), results["green_analysis"]
    finally:
        for path in (input_path, output_path):
            if os.path.exists(path):
                os.unlink(path)
