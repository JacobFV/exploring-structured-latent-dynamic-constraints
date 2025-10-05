#!/usr/bin/env python3
"""
HuggingFace Space Deployment Script

This script handles the deployment of essential files to HuggingFace Spaces.
"""

import os
import sys
from huggingface_hub import create_repo, HfApi


def deploy_to_huggingface():
    """Deploy essential files to HuggingFace Space"""

    token = os.environ.get('HF_TOKEN', '')
    if not token:
        print('HF_TOKEN is not set, skipping deployment')
        return False

    repo_id = 'exploring-structured-latent-dynamic-constraints'
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, repo_type='space', space_sdk='gradio')
        print('Created new HF Space')
    except Exception as e:
        print(f'Repo might already exist: {e}')

    # Upload essential files only
    print('Uploading essential files...')

    # Essential files for the space
    essential_files = [
        'pyproject.toml',
        'requirements.txt',
        'README.md'
    ]

    # Upload each essential file
    for file_path in essential_files:
        if os.path.exists(file_path):
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type='space',
                token=token
            )
            print(f'Uploaded {file_path}')

    # Create a simple app.py for the space
    app_content = '''import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

def analyze_model():
    # Placeholder for model analysis functionality
    return "Model analysis functionality would go here"

def finetune_model():
    # Placeholder for finetuning functionality
    return "Finetuning functionality would go here"

with gr.Blocks(title="Exploring Structured Latent Dynamic Constraints") as demo:
    gr.Markdown("# Exploring Structured Latent Dynamic Constraints")
    gr.Markdown("A space for analyzing and finetuning transformer models with structured latent constraints.")

    with gr.Tab("Model Analysis"):
        analyze_btn = gr.Button("Run Analysis")
        analysis_output = gr.Textbox(label="Analysis Results")

    with gr.Tab("Finetuning"):
        finetune_btn = gr.Button("Start Finetuning")
        finetune_output = gr.Textbox(label="Finetuning Results")

demo.launch()'''

    with open('app.py', 'w') as f:
        f.write(app_content)

    # Upload the app.py
    api.upload_file(
        path_or_fileobj='app.py',
        path_in_repo='app.py',
        repo_id=repo_id,
        repo_type='space',
        token=token
    )
    print('Uploaded app.py')

    print('Successfully deployed to HuggingFace!')
    return True


if __name__ == '__main__':
    success = deploy_to_huggingface()
    sys.exit(0 if success else 1)
