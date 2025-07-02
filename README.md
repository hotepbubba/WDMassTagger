# WD Mass Tagger

This repository provides scripts for tagging batches of images using Waifu Diffusion tagger models.

## Installation

Install the Python dependencies with pip:

```bash
python3 -m pip install -r requirements.txt
```

## Command Line Usage

Run the tagger on a directory of images:

```bash
python3 mass_tagger.py --targets-path /path/to/images [--recursive] [--dry-run] [--model-folder MODEL] [--tags-csv FILE] [--threshold FLOAT] [--batch-size INT] [--gpu-id INT]
```

- `--targets-path` (required): image file or folder containing images to tag.
- `--recursive`: search directories recursively.
- `--dry-run`: load images without running predictions.
- `--model-folder`: path or HuggingFace repo of the tagger model (default `networks/wd-v1-4-moat-tagger-v2`).
- `--tags-csv`: CSV file with labels (default `selected_tags.csv`).
- `--threshold`: prediction threshold (default `0.35`).
- `--batch-size`: batch size for inference (default `32`).
- `--gpu-id`: GPU index to run the model on (default `0`).

The script writes a `.txt` file next to each image containing the predicted tags.

## Gradio Interface

Launch the Gradio web UI with:

```bash
python3 gradio_app.py [--share]
```


Use the optional `--share` flag (or set the `SHARE=1` environment variable) to create a public link. This starts a local server where you can configure the options interactively and run the tagger through a browser interface.

This starts a local server where you can configure the options interactively and run the tagger through a browser interface.

In the *Batch Tag* tab, use the **Images** file uploader to select the pictures you want to tag. The selected files are processed on the server and the resulting tags are written next to each uploaded image.

