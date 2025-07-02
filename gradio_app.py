#!/usr/bin/env python3

import os
import argparse
import gradio as gr
from mass_tagger import tag_images, _load_model


def run(files, recursive, dry_run, model_folder, tags_csv, threshold, batch_size):
    file_paths = []
    if isinstance(files, list):
        for f in files:
            file_paths.append(f if isinstance(f, str) else f.name)
    elif files is not None:
        file_paths.append(files if isinstance(files, str) else files.name)

    if not file_paths:
        return "No files provided"

    with gr.Progress(track_tqdm=True) as progress:
        return tag_images(
            targets_path=file_paths,
            recursive=recursive,
            dry_run=dry_run,
            model_folder=model_folder,
            tags_csv=tags_csv,
            threshold=threshold,
            batch_size=batch_size,
            progress_tqdm=progress.tqdm,
        )


def run_single(upload, model_folder, tags_csv, threshold):
    """Tag a single uploaded image and return the tags."""
    if upload is None:
        return ""

    image_path = upload if isinstance(upload, str) else upload.name
    with gr.Progress(track_tqdm=True) as progress:
        tag_images(
            targets_path=image_path,
            recursive=False,
            dry_run=False,
            model_folder=model_folder,
            tags_csv=tags_csv,
            threshold=threshold,
            batch_size=1,
            progress_tqdm=progress.tqdm,
        )

    tags_file = os.path.splitext(image_path)[0] + ".txt"
    tags = ""
    if os.path.isfile(tags_file):
        with open(tags_file) as f:
            tags = f.read()
        os.remove(tags_file)

    if os.path.exists(image_path):
        os.remove(image_path)
    return tags



    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the Gradio interface publicly",
    )
    args = parser.parse_args()

    share_env = os.getenv("GRADIO_SHARE")
    if share_env is not None:
        share = share_env.lower() in ("1", "true", "yes")
    else:
        share = args.share

    parser = argparse.ArgumentParser(description="Launch the WD Mass Tagger UI")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link (can also set SHARE=1)",
    )
    args = parser.parse_args()

    share_flag = args.share or os.getenv("SHARE", "").lower() in ["1", "true", "yes"]


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
                    show_progress="full",
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
                    show_progress="full",
                )

    demo.queue()

    demo.launch(share=share)

    demo.launch(share=share_flag)



if __name__ == "__main__":
    main()
