#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================================="
echo "  Node.js, npm, and Qwen Code Installer"
echo "  This script must be run as root."
echo "========================================================="
echo

# --- 1. Remove Existing Node.js and npm ---
echo "[Step 1/4] Checking for and removing existing Node.js installations..."

# Detect package manager
if command -v apt-get &> /dev/null; then
    echo "Detected APT package manager (Debian/Ubuntu)."
    # Using 'purge' to also remove configuration files
    apt-get purge -y nodejs npm &> /dev/null || true
    apt-get autoremove -y &> /dev/null || true
    echo "Old Node.js/npm packages removed."

elif command -v dnf &> /dev/null; then
    echo "Detected DNF package manager (Fedora/RHEL/CentOS 8+)."
    dnf remove -y nodejs npm &> /dev/null || true
    echo "Old Node.js/npm packages removed."

elif command -v yum &> /dev/null; then
    echo "Detected YUM package manager (CentOS 7)."
    yum remove -y nodejs npm &> /dev/null || true
    echo "Old Node.js/npm packages removed."
else
    echo "Warning: Could not detect a supported package manager (apt, dnf, yum). Skipping removal."
fi
echo "---------------------------------------------------------"


# --- 2. Install Latest Node.js (LTS) without curl ---
echo "[Step 2/4] Installing the latest LTS version of Node.js using wget..."

# Using the official NodeSource repository for the latest versions
# We use wget as a direct replacement for curl
if command -v apt-get &> /dev/null; then
    # Ensure dependencies for adding a repo are present
    apt-get update
    apt-get install -y ca-certificates gnupg
    # Download and execute the NodeSource setup script
    wget -qO- https://deb.nodesource.com/setup_lts.x | bash -
    apt-get install -y nodejs

elif command -v dnf &> /dev/null || command -v yum &> /dev/null; then
    # Download and execute the NodeSource setup script
    wget -qO- https://rpm.nodesource.com/setup_lts.x | bash -
    # DNF/YUM will be used automatically by the 'nodejs' package name
    yum install -y nodejs
else
    echo "Error: Cannot install Node.js. No supported package manager found."
    exit 1
fi

echo "Node.js and npm installed successfully."
node -v
npm -v
echo "---------------------------------------------------------"


# --- 3. Install Qwen Code ---
echo "[Step 3/4] Installing @qwen-code/qwen-code globally via npm..."
# The --unsafe-perm flag can be necessary when running npm as root
npm install -g @qwen-code/qwen-code@latest --unsafe-perm
echo "---------------------------------------------------------"


# --- 4. Verify Qwen Code Installation ---
echo "[Step 4/4] Verifying Qwen Code installation..."
# The 'hash -r' command clears the command lookup path cache
hash -r
qwen --version
echo "---------------------------------------------------------"

echo
echo "========================================================="
echo "  Setup Complete!"
echo "  Node.js, npm, and Qwen Code are now installed."
echo "========================================================="