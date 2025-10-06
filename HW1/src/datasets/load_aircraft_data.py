from huggingface_hub import snapshot_download
import os


def load_aircraft_dataset(path, num_folders):

    repo_id = "Voxel51/FGVC-Aircraft"
    patterns = [f"data/data_{i}/*" for i in range(num_folders)]
    os.makedirs(path, exist_ok=True)

    dataset_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        token="",
        local_dir=path
    )