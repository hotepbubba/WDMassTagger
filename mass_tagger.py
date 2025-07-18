import argparse
import os
from pathlib import Path

KAOMOJIS = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]

import pandas as pd
from tqdm import tqdm

# Very important very obscure flags, make things go brrrr
# They incur high startup times though, so only useful for big jobs
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

# Reduce logging
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
from huggingface_hub import snapshot_download

_model_cache = {}
_model_paths = {}
_tags_cache = {}


def _load_model(model_folder):
    """Load and cache a TensorFlow model."""
    cached = _model_cache.get(model_folder)
    if cached is not None:
        return cached

    path = model_folder
    if not Path(model_folder).exists():
        print(f"Downloading model '{model_folder}' from Hugging Face...")
        path = snapshot_download(repo_id=model_folder)

    model = tf.keras.models.load_model(path)
    _, height, width, _ = model.inputs[0].shape
    cached = (model, int(height), int(width), path)
    _model_cache[model_folder] = cached
    _model_paths[model_folder] = path
    return cached


def _load_tags(labels_file):
    """Load and cache tags dataframe."""
    if labels_file in _tags_cache:
        return _tags_cache[labels_file]

    df = pd.read_csv(labels_file)
    df["sanitized_name"] = df["name"].map(
        lambda x: x.replace("_", " ") if x not in KAOMOJIS else x
    )
    _tags_cache[labels_file] = df
    return df


from Generator.TFDataReader import DataGenerator

# Stop TF from hogging all of the VRAM
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def process_images(filepaths, images, model, tags_df, threshold, device="/CPU:0"):
    @tf.function
    def pred_model(x):
        return model(x, training=False)

    with tf.device(device):
        preds = pred_model(images).numpy()

    for image_path, pred in zip(filepaths, preds):
        image_path = image_path.numpy().decode("utf-8")

        tags_df["preds"] = pred
        general_tags = tags_df[tags_df["category"] == 0]
        chosen_tags = general_tags[general_tags["preds"] > threshold]
        chosen_tags = chosen_tags.sort_values(by="preds", ascending=False)
        tags_names = chosen_tags["sanitized_name"]
        tags_string = ", ".join(tags_names)
        with open(Path(image_path).with_suffix(".txt"), "w") as f:
            f.write(tags_string)


def dataset_diagnostic(filepaths, images):
    lines = []
    for image_path in filepaths:
        image_path = image_path.numpy().decode("utf-8")
        lines.append(f"{image_path}\n")
    with open("dry_run_read.txt", "a") as f:
        f.writelines(lines)


def tag_images(
    targets_path,
    recursive=False,
    dry_run=False,
    model_folder="networks/wd-v1-4-moat-tagger-v2",
    tags_csv="selected_tags.csv",
    threshold=0.35,
    batch_size=32,
    gpu_id=0,
    progress_tqdm=tqdm,
):
    """Tag images from a path or list using a WD14 tagger model."""

    glob_pattern = "**/*" if recursive else "*"

    image_extensions = [".jpeg", ".jpg", ".png", ".webp"]

    if isinstance(targets_path, (list, tuple)):
        images_list = [
            str(Path(p).resolve())
            for p in targets_path
            if Path(p).suffix.lower() in image_extensions
        ]
    else:
        targets_path = Path(targets_path)
        if targets_path.is_file():
            images_list = (
                [str(targets_path.resolve())]
                if targets_path.suffix.lower() in image_extensions
                else []
            )
        else:
            images_list = [
                str(p.resolve())
                for p in targets_path.glob(glob_pattern)
                if p.suffix.lower() in image_extensions
            ]

    model, height, width, model_path = _load_model(model_folder)

    labels_file = tags_csv
    if not Path(labels_file).is_file():
        candidate = Path(model_path) / labels_file
        if candidate.is_file():
            labels_file = str(candidate)

    # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368

    tags_df = _load_tags(labels_file)

    use_gpu = tf.config.list_physical_devices("GPU") and gpu_id is not None
    device_name = f"/GPU:{gpu_id}" if use_gpu else "/CPU:0"

    if not dry_run:
        process_func = lambda paths, imgs: process_images(
            paths, imgs, model, tags_df, threshold, device_name
        )
    else:
        height, width = 224, 224
        process_func = dataset_diagnostic

        scheduled = [f"{image_path}\n" for image_path in images_list]

        # Truncate the file from previous runs
        open("dry_run_read.txt", "w").close()
        with open("dry_run_scheduled.txt", "w") as f:
            f.writelines(scheduled)

    generator = DataGenerator(
        file_list=images_list, target_size=height, batch_size=batch_size
    ).genDS()

    for filepaths, images in progress_tqdm(generator):
        process_func(filepaths, images)

    return f"Processed {len(images_list)} images"


def tag_single_image(
    image_path,
    model_folder="networks/wd-v1-4-moat-tagger-v2",
    tags_csv="selected_tags.csv",
    threshold=0.35,
    gpu_id=0,
):
    """Return predicted tags for a single image."""

    image_path = Path(image_path)

    model, height, width, model_path = _load_model(model_folder)

    labels_file = tags_csv
    if not Path(labels_file).is_file():
        candidate = Path(model_path) / labels_file
        if candidate.is_file():
            labels_file = str(candidate)

    tags_df = _load_tags(labels_file)

    use_gpu = tf.config.list_physical_devices("GPU") and gpu_id is not None
    device_name = f"/GPU:{gpu_id}" if use_gpu else "/CPU:0"

    with tf.device(device_name):
        pass  # model already loaded

    generator = DataGenerator(
        file_list=[str(image_path.resolve())],
        target_size=height,
        batch_size=1,
    ).genDS()

    for filepaths, images in generator:
        with tf.device(device_name):
            pred = model(images, training=False).numpy()[0]

    tags_df["preds"] = pred
    general_tags = tags_df[tags_df["category"] == 0]
    chosen_tags = general_tags[general_tags["preds"] > threshold]
    chosen_tags = chosen_tags.sort_values(by="preds", ascending=False)
    tags_names = chosen_tags["sanitized_name"]
    return ", ".join(tags_names)


parser = argparse.ArgumentParser(description="Mass tag a set of images")

# Images arguments
parser.add_argument(
    "--targets-path",
    required=True,
    help="Image file or folder with the images to tag",
)
parser.add_argument(
    "--recursive",
    action="store_true",
    help="Recurse directories when looking for images",
)
parser.set_defaults(recursive=False)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Load the images without running predictions",
)
parser.set_defaults(dry_run=False)

# Model arguments
parser.add_argument("--model-folder", default="networks/wd-v1-4-moat-tagger-v2")
parser.add_argument("--tags-csv", default="selected_tags.csv")
parser.add_argument(
    "--threshold",
    default=0.35,
    type=float,
    help="Predictions threshold",
)
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Batch size",
)
parser.add_argument(
    "--gpu-id",
    default=0,
    type=int,
    help="GPU index to use for TensorFlow operations",
)


def main():
    args = parser.parse_args()
    tag_images(
        targets_path=args.targets_path,
        recursive=args.recursive,
        dry_run=args.dry_run,
        model_folder=args.model_folder,
        tags_csv=args.tags_csv,
        threshold=args.threshold,
        batch_size=args.batch_size,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
