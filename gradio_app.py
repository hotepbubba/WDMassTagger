#!/usr/bin/env python3

import os
import gradio as gr
from mass_tagger import tag_images, _load_model


def run(
    files,
    recursive,
    dry_run,
    model_folder,
    tags_csv,
    threshold,
    batch_size,
):
    file_paths = []
    if isinstance(files, list):
        for f in files:
            file_paths.append(f if isinstance(f, str) else f.name)
    elif files is not None:
        file_paths.append(files if isinstance(files, str) else files.name)

    if not file_paths:
        return "No files provided"

    return tag_images(
        targets_path=file_paths,
        recursive=False,
        dry_run=dry_run,
        model_folder=model_folder,
        tags_csv=tags_csv,
        threshold=threshold,
        batch_size=batch_size,
    )


def run_single(upload, model_folder, tags_csv, threshold):
    """Tag a single uploaded image and return the tags."""
    if upload is None:
        return ""

    image_path = upload if isinstance(upload, str) else upload.name
    tag_images(
        targets_path=image_path,
        recursive=False,
        dry_run=False,
        model_folder=model_folder,
        tags_csv=tags_csv,
        threshold=threshold,
        batch_size=1,
    )

    tags_file = os.path.splitext(image_path)[0] + ".txt"
    if os.path.isfile(tags_file):
        with open(tags_file) as f:
            tags = f.read()
        os.remove(tags_file)
    else:
        tags = ""

    if os.path.exists(image_path):
        os.remove(image_path)
    return tags


def main():
    models = [
        "SmilingWolf/wd-eva02-large-tagger-v3",
        "SmilingWolf/wd-vit-large-tagger-v3",
        "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
        "SmilingWolf/wd-vit-tagger-v3",
        "SmilingWolf/wd-swinv2-tagger-v3",
        "SmilingWolf/wd-convnext-tagger-v3",
        "SmilingWolf/wd-v1-4-moat-tagger-v2",
        "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        "SmilingWolf/wd-v1-4-vit-tagger-v2",
        "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
        "SmilingWolf/wd-v1-4-convnext-tagger",
        "SmilingWolf/wd-v1-4-vit-tagger",
    ]

    # Preload the default model so the first request is responsive
    default_model = "SmilingWolf/wd-v1-4-moat-tagger-v2"
    _load_model(default_model)

    with gr.Blocks() as demo:
        gr.Markdown("# WD Mass Tagger")
        with gr.Tabs():
            with gr.TabItem("Batch Tag"):
                targets = gr.Files(label="Images")
                recursive = gr.Checkbox(label="Recursive")
                dry_run = gr.Checkbox(label="Dry Run")
                model_folder = gr.Dropdown(
                    label="Model Folder or HuggingFace repo",
                    choices=models,
                    value=default_model,
                )
                tags_csv = gr.Textbox(label="Tags CSV", value="selected_tags.csv")
                threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.35, label="Threshold"
                )
                batch_size = gr.Number(value=32, label="Batch Size", precision=0)
                out = gr.Textbox(label="Status")
                run_button = gr.Button("Run")
                run_button.click(
                    run,
                    [
                        targets,
                        recursive,
                        dry_run,
                        model_folder,
                        tags_csv,
                        threshold,
                        batch_size,
                    ],
                    out,
                )

            with gr.TabItem("Single Image"):
                image = gr.Image(label="Image", type="filepath")
                model_folder_s = gr.Dropdown(
                    label="Model Folder or HuggingFace repo",
                    choices=models,
                    value=default_model,
                )
                tags_csv_s = gr.Textbox(label="Tags CSV", value="selected_tags.csv")
                threshold_s = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.35, label="Threshold"
                )
                out_s = gr.Textbox(label="Tags")
                run_button_s = gr.Button("Run")
                run_button_s.click(
                    run_single,
                    [image, model_folder_s, tags_csv_s, threshold_s],
                    out_s,
                )
    demo.launch(share=True)


if __name__ == "__main__":
    main()
