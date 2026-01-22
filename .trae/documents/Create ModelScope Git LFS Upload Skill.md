## ModelScope Git LFS Upload Skill

I'll create a Trae skill that provides step-by-step instructions for uploading large files to ModelScope using Git LFS, with specific guidance for the provided test case.

### Skill Structure

1. **YAML Frontmatter**:
   - name: modelscope-lfs-upload
   - description: Guide for uploading large files to ModelScope using Git LFS
   - license: MIT

2. **Markdown Body**:
   - Introduction to ModelScope and Git LFS
   - Prerequisites (Git, Git LFS installation)
   - Step-by-step upload process
   - Specific instructions for the test case
   - Troubleshooting tips

3. **Usage Example**:
   ```
   @modelscope-lfs-upload
   ```

### Files to Create

- `/data1/syliu/ixi_mcx_2025/.trae/skills/modelscope-lfs-upload/SKILL.md`

### Key Instructions to Include

1. Installing Git and Git LFS
2. Creating a ModelScope repository
3. Cloning with OAuth token
4. Configuring Git LFS for large files
5. Adding files
6. Pushing to ModelScope
7. Specific commands for the test case

### Test Case Integration

The skill will include specific instructions for the provided test case:
- Repository: `https://www.modelscope.cn/datasets/syliubupt2025/ixi_bupt_mcx_shiyu`
- Clone URL: `https://oauth2:ms-c1b331d2-38ea-47f8-960b-8f3ac78f59ba@www.modelscope.cn/datasets/syliubupt2025/ixi_bupt_mcx_shiyu.git`

This skill will help users upload large files to ModelScope efficiently using Git LFS, with clear guidance for the specific test case provided.