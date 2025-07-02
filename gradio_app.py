#!/usr/bin/env python3

import gradio as gr
from mass_tagger import tag_images


def run(
    targets_path,
    recursive,
    dry_run,
    model_folder,
    tags_csv,
    threshold,
    batch_size,
):
    return tag_images(
        targets_path=targets_path,
        recursive=recursive,
        dry_run=dry_run,
        model_folder=model_folder,
        tags_csv=tags_csv,
        threshold=threshold,
        batch_size=batch_size,
    )


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# WD Mass Tagger")
        targets = gr.Textbox(label="Targets Path")
        recursive = gr.Checkbox(label="Recursive")
        dry_run = gr.Checkbox(label="Dry Run")
        model_folder = gr.Textbox(
            label="Model Folder or HuggingFace repo",
            value="networks/wd-v1-4-moat-tagger-v2",
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
            [targets, recursive, dry_run, model_folder, tags_csv, threshold, batch_size],
            out,
        )
    demo.launch(share=True)


if __name__ == "__main__":
    main()
