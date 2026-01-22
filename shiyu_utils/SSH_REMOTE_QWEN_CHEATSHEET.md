https://ecs.console.aliyun.com/server/i-bp1135m5grcrtmn49lzh/detail?regionId=cn-hangzhou&serverLiteToken=1756495473566












Of course. Here is a cheat sheet summarizing how to install Node.js without `curl` and how to keep a shell session running after you disconnect.

### Remote Server Survival Cheat Sheet

---

#### 1. Install Node.js LTS on a New Debian/Ubuntu Server

*Assumes you are running as `root` or using `sudo`.*

**Step 1: Add NodeSource Repository & GPG Key**
```bash
# Create keyring directory
mkdir -p /etc/apt/keyrings

# Download and add NodeSource GPG key using wget
wget -qO- https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
```

**Step 2: Add the Repository to sources.list**
```bash
# Set the desired Node.js major version (e.g., 20 for LTS)
NODE_MAJOR=20

# Create the source file
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list
```

**Step 3: Install Node.js**
```bash
# Update package lists and install nodejs
apt update
apt install nodejs -y
```

**Step 4: Verify Installation**
```bash
node -v
npm -v
```

---

#### 2. Keep Shells & Commands Running After Disconnecting

##### Method A: `tmux` (Modern & Recommended)

`tmux` creates persistent, interactive shell sessions you can detach from and re-attach to later.

| Action                  | Command                                 | Key Combination (inside session) |
| :---------------------- | :-------------------------------------- | :------------------------------- |
| **Start a new session** | `tmux new -s <session_name>`            | N/A                              |
| **List sessions**       | `tmux ls`                               | N/A                              |
| **Detach from session** | *N/A*                                   | **`Ctrl+b`** then **`d`**        |
| **Re-attach to session**| `tmux attach -t <session_name>`         | N/A                              |
| **Kill a session**      | `tmux kill-session -t <session_name>`   | Type `exit` or **`Ctrl+d`**      |

**Example Workflow:**
1.  `tmux new -s web_server`      *(Start a new session named "web_server")*
2.  Run your long process (e.g., `npm start`)
3.  Press **`Ctrl+b`**, then **`d`**   *(Detach and go back to your main shell)*
4.  (Disconnect from SSH and reconnect later)
5.  `tmux attach -t web_server`   *(Get back into your running session)*

---

##### Method B: `screen` (Classic & Widely Available)

`screen` is the classic alternative to `tmux`.

| Action                  | Command                             | Key Combination (inside session) |
| :---------------------- | :---------------------------------- | :------------------------------- |
| **Start a new session** | `screen -S <session_name>`          | N/A                              |
| **List sessions**       | `screen -ls`                        | N/A                              |
| **Detach from session** | *N/A*                               | **`Ctrl+a`** then **`d`**        |
| **Re-attach to session**| `screen -r <session_name>`          | N/A                              |
| **Kill a session**      | *Attach and exit*                   | Type `exit` or **`Ctrl+d`**      |

---

##### Method C: `nohup` (For Single, Non-interactive Commands)

Use `nohup` to run a command that will ignore the "hangup" signal when you disconnect. It's best for "fire-and-forget" scripts.

**Basic Usage (output goes to `nohup.out`):**
```bash
nohup <your_command> &
```

**Advanced Usage (redirect output and errors to a log file):**
```bash
nohup <your_command> > my_output.log 2>&1 &
```
*   `>`: Redirects standard output.
*   `2>&1`: Redirects standard error to the same place.
*   `&`: Puts the process in the background.

**To find the process later:**
```bash
ps aux | grep <your_command>
```
**To stop the process:**
```bash
kill <process_id>
```