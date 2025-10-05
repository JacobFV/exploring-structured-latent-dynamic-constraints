#!/usr/bin/env python3
"""
HuggingFace Space Deployment Script

This script handles the deployment of essential files to HuggingFace Spaces.
"""

import os
import sys
import shutil
from pathlib import Path

# Add parent directory to path so we can import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import HfApi


def deploy_to_huggingface():
    """Deploy essential files to HuggingFace Space"""

    token = os.environ.get('HF_TOKEN', '')
    if not token:
        print('HF_TOKEN is not set, skipping deployment')
        return False

    repo_id = 'exploring-structured-latent-dynamic-constraints'
    api = HfApi(token=token)

    print(f'Deploying to existing HF Space: {repo_id}')

    # Upload essential files only
    print('Uploading essential files...')

    # Essential files for the space (from project root)
    project_root = Path(__file__).parent.parent
    essential_files = [
        'pyproject.toml',
        'README.md'
    ]

    # Upload each essential file
    for file_path in essential_files:
        full_path = project_root / file_path
        if full_path.exists():
            api.upload_file(
                path_or_fileobj=str(full_path),
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type='space',
                token=token
            )
            print(f'Uploaded {file_path}')
        else:
            print(f'Warning: {file_path} not found')

    # Copy gradio app to root for HF Spaces
    gradio_app_path = project_root / 'gradio' / 'space_app.py'
    root_app_path = project_root / 'app.py'
    if gradio_app_path.exists():
        shutil.copy2(gradio_app_path, root_app_path)
        api.upload_file(
            path_or_fileobj=str(root_app_path),
            path_in_repo='app.py',
            repo_id=repo_id,
            repo_type='space',
            token=token
        )
        print('Uploaded app.py (from gradio/space_app.py)')
        # Optionally remove the temporary app.py after upload
        root_app_path.unlink()
    else:
        print('Warning: gradio/space_app.py not found')

    print('Successfully deployed to HuggingFace!')
    return True


if __name__ == '__main__':
    success = deploy_to_huggingface()
    sys.exit(0 if success else 1)
