from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm


MODEL_BASE_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    file_name: str

    @property
    def url(self) -> str:
        return f"{MODEL_BASE_URL}/{self.file_name}"


MODEL_SPECS: dict[str, ModelSpec] = {
    "ssd_mobilenet_v2": ModelSpec(
        name="ssd_mobilenet_v2",
        file_name="ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz",
    ),
    "efficientdet_d0": ModelSpec(
        name="efficientdet_d0",
        file_name="efficientdet_d0_coco17_tpu-32.tar.gz",
    ),
    "faster_rcnn_resnet50": ModelSpec(
        name="faster_rcnn_resnet50",
        file_name="faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz",
    ),
}


def available_models() -> list[str]:
    return sorted(MODEL_SPECS)


def get_model_dir(model_name: str, models_dir: Path) -> Path:
    spec = _get_spec(model_name)
    return models_dir / spec.file_name.removesuffix(".tar.gz") / "saved_model"


def ensure_model(model_name: str, models_dir: Path) -> Path:
    spec = _get_spec(model_name)
    saved_model_dir = get_model_dir(model_name, models_dir)
    if (saved_model_dir / "saved_model.pb").exists():
        return saved_model_dir

    models_dir.mkdir(parents=True, exist_ok=True)
    archive_path = models_dir / spec.file_name
    if not archive_path.exists():
        _download_file(spec.url, archive_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract(archive, models_dir)

    if not (saved_model_dir / "saved_model.pb").exists():
        raise FileNotFoundError(
            f"Downloaded model did not contain expected SavedModel: {saved_model_dir}"
        )
    return saved_model_dir


def _get_spec(model_name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_name]
    except KeyError as exc:
        choices = ", ".join(available_models())
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {choices}") from exc


def _download_file(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with destination.open("wb") as file_obj, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=destination.name,
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file_obj.write(chunk)
                progress.update(len(chunk))


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if not str(member_path).startswith(str(destination)):
            raise ValueError(f"Unsafe archive path: {member.name}")
    archive.extractall(destination)
