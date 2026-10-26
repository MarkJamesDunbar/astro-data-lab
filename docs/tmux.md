### Using tmux

`tmux` is a headless terminal for the homelab and allows processes to continue running on the homelab after disconnecting from SSH or closing VS Code. I use this before I've set up a proper scheduler.

This is useful for lengthy operations such as MAST inventory synchronisation, dataset downloads and model training.

**Common commands:**

| Command | Purpose |
|---|---|
| `tmux new -s job-name` | Create a named terminal session |
| `tmux ls` | List existing sessions |
| `tmux attach -t job-name` | Reconnect to a session |
| `Ctrl+B`, then `D` | Detach without stopping the process |
| `tmux kill-session -t job-name` | Close a session and its processes |
| `exit` | Close the current shell/session |

**Example workflow:**

1. Create a session with `tmux new -s mast-backfill`.
2. Start the Python ingestion process inside it.
3. Detach using `Ctrl+B`, then `D`.
4. Reconnect later with `tmux attach -t mast-backfill`.
5. Once the process has finished, close the session.

**Note:** tmux survives SSH disconnections but not server reboots. For recurring unattended jobs, we'll use `systemd` services and timers instead.