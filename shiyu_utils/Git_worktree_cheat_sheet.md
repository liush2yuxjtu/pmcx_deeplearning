好的，没问题。这是您刚刚学习的 Git worktree 并行工作流的中文版速查表。

### Git Worktree 并行工作流速查表

**它是什么？**
这是一种 Git 工作方式，允许你在不同的目录中同时检出（check out）并处理同一个仓库的多个分支。所有这些目录都链接到同一个 `.git` 仓库。它非常适合并行处理多个任务、测试新功能或修复 Bug，而不会干扰你的主要工作。

---

#### **核心命令**

| 命令 | 描述 |
| :--- | :--- |
| `git worktree list` | 显示所有活动的工作区及其对应的分支。 |
| `git worktree add <路径> -b <新分支名>` | 在指定 `<路径>` 创建一个**新分支**和一个**新的工作区**。 |
| `git worktree add <路径> <已有分支名>` | 从一个**已有的分支**在 `<路径>` 创建一个新的工作区。 |
| `git worktree remove <路径>` | 移除指定 `<路径>` 的工作区。 |
| `git worktree prune` | 清理 `.git` 目录中残留的、无效的工作区数据。 |
| `git branch -d <分支名>`| 在分支合并后，删除该特性分支。 |

---

#### **标准并行工作流**

**1. 设置你的项目**
   - 首先，进入你的主项目目录。
   - 最佳实践是为所有工作区创建一个专门的文件夹。
     ```bash
     mkdir WorkTrees
     ```

**2. 为并行任务创建工作区**
   - 为每个独立的任务创建一个新的工作区和新分支。
     ```bash
     # 语法: git worktree add <工作区路径> -b <新分支名>
     git worktree add ./WorkTrees/feature-A -b feature-A-branch
     git worktree add ./WorkTrees/feature-B -b feature-B-branch
     ```

**3. 在不同任务上独立工作**
   - 为每个工作区打开一个**独立的终端**。
   - `cd` 进入特定的工作区目录以处理该任务。
     ```bash
     # 在终端 1 中:
     cd ./WorkTrees/feature-A
     # ... 修改代码, git add, git commit ...
     
     # 在终端 2 中:
     cd ./WorkTrees/feature-B
     # ... 修改代码, git add, git commit ...
     ```

**4. 合并与清理**
   - 当所有特性分支的工作都完成后，返回你的主项目目录。
     ```bash
     # 回到项目根目录
     cd /path/to/your/main/project
     
     # 切换到主分支
     git switch main
     
     # 合并已完成的特性分支
     git merge feature-A-branch
     git merge feature-B-branch
     
     # 移除不再需要的工作区
     git worktree remove ./WorkTrees/feature-A
     git worktree remove ./WorkTrees/feature-B

     # (可选) 删除已经合并的特性分支
     git branch -d feature-A-branch
     git branch -d feature-B-branch
     ```

---

#### **关键提示**

*   **隔离是关键：** 最大的好处是在一个工作区中的修改不会影响其他工作区。
*   **一个分支，一个工作区：** 避免在多个工作区中检出同一个分支，以免造成混淆。
*   **共享仓库历史：** 所有工作区共享同一个中央 `.git` 历史记录，因此非常高效。只有工作文件是分开的。