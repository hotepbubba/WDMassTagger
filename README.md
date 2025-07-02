# WD Mass Tagger

This repository provides scripts for tagging batches of images using Waifu Diffusion tagger models.

## Installation

Run the setup script to create a virtual environment and install the required
packages:

```bash
./install.sh
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


Passing `--share` or setting the environment variable `GRADIO_SHARE=1` will share the interface publicly via Gradio.

In the *Batch Tag* tab, use the **Images** file uploader to select the pictures you want to tag. The selected files are processed on the server and the resulting tags are written next to each uploaded image.

## Launch

Use the helper script to start the Gradio interface from the virtual environment:

```bash
./launch.sh [--share]
```

Any arguments passed to `launch.sh` are forwarded to `gradio_app.py`.

## Desktop Shortcut

Copy the provided `wdmass-tagger.desktop` file to `~/.local/share/applications` so that WD Mass Tagger appears in your application menu.

Place your own icon at `assets/icon.png` if you want the shortcut to display one.

