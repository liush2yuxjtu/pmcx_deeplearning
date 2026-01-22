---
name: modelscope-lfs-upload
description: Guide for uploading large files to ModelScope using Git LFS. Use when users need to upload large files, datasets, or models to ModelScope.
license: MIT
---

# ModelScope Git LFS Upload Guide

## About This Skill

This skill provides step-by-step instructions for uploading large files to ModelScope using Git LFS, including specific guidance for the test case provided.

## When to Use

Use this skill when:
- Uploading large files (>100MB) to ModelScope
- Uploading datasets or models to ModelScope
- Following the test case requirements

## Test Case Instructions

For the test case, follow these specific steps:

1. **Clone the repository** using the provided OAuth URL
2. **Configure Git LFS** for large file support
3. **Add files** to the repository
4. **Push** to ModelScope

## Prerequisites

- Git and Git LFS installed
- ModelScope account
- Git access token from ModelScope

## Step-by-Step Guide

### 1. Install Git LFS

```bash
git lfs install
```

### 2. Clone the Repository

Use the provided OAuth URL:

```bash
git clone https://oauth2:ms-c1b331d2-38ea-47f8-960b-8f3ac78f59ba@www.modelscope.cn/datasets/syliubupt2025/ixi_bupt_mcx_shiyu.git
```

### 3. Configure Git LFS Tracking

```bash
# Track all large files
git lfs track "*.pt"  # For PyTorch models
git lfs track "*.pth"  # For PyTorch checkpoints
git lfs track "*.ckpt" # For general checkpoints
git lfs track "*.bin"  # For binary files
git lfs track "*.zip"  # For compressed files
git lfs track "*.tar.gz" # For compressed archives

# Track entire directories (optional)
git lfs track "data/**"  # Track all files in data directory
git lfs track "models/**" # Track all files in models directory
```

### 4. Add and Commit Files

```bash
# Add files
git add .

# Commit changes
git commit -m "Add large files"
```

### 5. Push to ModelScope

```bash
git push origin main
```

## Test Case Specific Instructions

For the provided test case:

1. Clone the repository:
   ```bash
   git clone https://oauth2:ms-c1b331d2-38ea-47f8-960b-8f3ac78f59ba@www.modelscope.cn/datasets/syliubupt2025/ixi_bupt_mcx_shiyu.git
   ```

2. Navigate to the repository:
   ```bash
   cd ixi_bupt_mcx_shiyu
   ```

3. Install Git LFS:
   ```bash
   git lfs install
   ```

4. Configure Git LFS tracking for your large files:
   ```bash
   git lfs track "*.pt" "*.pth" "*.ckpt" "*.bin" "*.zip" "*.tar.gz"
   ```

5. Add your large files to the repository:
   ```bash
   # Copy your large files here
   git add .
   ```

6. Commit and push:
   ```bash
   git commit -m "Add large files"
   git push origin main
   ```

## Troubleshooting

### Git LFS Authentication Issues

If you encounter authentication issues, ensure:
- Your OAuth token is correct and has the necessary permissions
- You're using the full OAuth URL format
- Git is configured with your credentials

### Large File Pushing Issues

- Increase Git buffer size if pushing very large files:
  ```bash
  git config http.postBuffer 524288000  # 500MB buffer
  ```

- Check network connectivity if pushes fail
- Verify Git LFS is properly installed

## Best Practices

- **Track only large files** with Git LFS (typically >100MB)
- **Use .gitattributes** to define LFS tracking rules
- **Test with small files first** before uploading large datasets
- **Monitor push progress** for large uploads
- **Use descriptive commit messages** for better version control

## Output Example

When successful, you'll see output like:

```
Uploading LFS objects: 100% (3/3), 1.2 GB | 10.5 MB/s, done.
Enumerating objects: 15, done.
Counting objects: 100% (15/15), done.
Delta compression using up to 8 threads.
Compressing objects: 100% (12/12), done.
Writing objects: 100% (15/15), 2.15 KiB | 2.15 KiB/s, done.
Total 15 (delta 3), reused 0 (delta 0), pack-reused 0
To https://www.modelscope.cn/datasets/syliubupt2025/ixi_bupt_mcx_shiyu.git
   1a2b3c4..5d6e7f8  main -> main
```