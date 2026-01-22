#!/bin/bash

# Script to install and setup Git LFS for the project

# Check if Git LFS is already installed
if command -v git-lfs &> /dev/null
then
    echo "Git LFS is already installed"
    git lfs version
else
    echo "Installing Git LFS..."
    
    # Download and install Git LFS based on the OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # For Linux
        curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
        sudo apt-get install git-lfs
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # For macOS
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
        brew install git-lfs
    else
        echo "Unsupported OS. Please install Git LFS manually."
        echo "Visit https://git-lfs.github.com/ for installation instructions."
        exit 1
    fi
fi

# Initialize Git LFS in the repository
git lfs install

# Setup the necessary file patterns for Git LFS
git lfs track "models/*.pt"
git lfs track "checkpoints_*/**/*.pt"

echo "Git LFS setup complete!"
echo "Please run 'git add .gitattributes' to add the LFS configuration to your repository."