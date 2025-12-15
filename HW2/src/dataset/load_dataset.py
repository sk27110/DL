import os
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path
import json
import logging

logger = logging.getLogger("load_data")


def download_dataset(dataset_name: str, download_dir: str, conf_dir: str = "./conf") -> str:
    """
    Скачивает датасет с Kaggle в указанную директорию с отображением прогресса.
    Использует kaggle.json из conf_dir.

    Args:
        dataset_name (str): "username/dataset"
        download_dir (str): папка для сохранения датасета
        conf_dir (str): папка, где лежит kaggle.json

    Returns:
        str: путь к распакованной папке
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    kaggle_json_path = Path(conf_dir) / "kaggle.json"
    if not kaggle_json_path.exists():
        logger.error(f"❌ Не найден {kaggle_json_path}")
        raise FileNotFoundError(f"❌ Не найден {kaggle_json_path}")

    # Загружаем токен
    with open(kaggle_json_path, "r") as f:
        creds = json.load(f)
    username = creds["username"]
    key = creds["key"]

    # Настраиваем пути
    dataset_slug = dataset_name.split("/")[-1]
    extracted_path = download_dir / dataset_slug
    zip_path = download_dir / f"{dataset_slug}.zip"

    # --- 🔍 Проверка: уже скачано ---
    if extracted_path.exists() and any(extracted_path.iterdir()):
        logger.info(f"✅ Датасет уже существует по пути: {extracted_path}")
        return str(extracted_path)

    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_name}"
    logger.info(f"📦 Скачиваем {dataset_name} в {zip_path} ...")

    # --- ⬇️ Скачиваем с прогрессом ---
    with requests.get(url, auth=(username, key), stream=True) as r:
        if r.status_code != 200:
            logger.error(f"Ошибка загрузки: {r.status_code}, {r.text[:200]}")
            raise RuntimeError(f"Ошибка загрузки: {r.status_code}, {r.text[:200]}")

        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        with open(zip_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="⬇️ Downloading"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=block_size):
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"✅ Загружено: {zip_path}")

    # --- 📂 Распаковываем ---
    extracted_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Распаковка в {extracted_path} ...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="📦 Extracting"):
            zip_ref.extract(member, extracted_path)

    logger.info(f"✅ Распаковано в: {extracted_path}")

    # --- 🧹 Удаляем zip ---
    zip_path.unlink(missing_ok=True)

    return str(extracted_path)
