import os
import sys
import time
import platform
import subprocess
import base64
from datetime import datetime
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import git
import psutil
import numpy as np
import cv2

try:
    import torch
except ImportError:
    torch = None

class ProjectTracker:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.resolve()
        self.notebook_path = self.root_dir / "notebooks" / "Project_Progress.ipynb"
        self.output_dirs = ["outputs", "debug", "results"]
        self.video_dir = Path("outputs/videos")
        
        # Ensure directory exists
        self.notebook_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.repo = git.Repo(self.root_dir, search_parent_directories=True)
        except:
            self.repo = None

    def get_timestamp(self):
        now = datetime.now()
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
            "iso": now.isoformat()
        }

    def get_git_info(self):
        if not self.repo:
            return "Git repository not initialized."
        
        try:
            head = self.repo.head.reference
            commit = head.commit
            status = self.repo.git.status()
            untracked = self.repo.untracked_files
            modified = [item.a_path for item in self.repo.index.diff(None)]
            total_commits = len(list(self.repo.iter_commits()))
            
            return {
                "branch": self.repo.active_branch.name,
                "hash": commit.hexsha,
                "message": commit.message.strip(),
                "author": commit.author.name,
                "date": datetime.fromtimestamp(commit.authored_date).strftime('%Y-%m-%d %H:%M:%S'),
                "total_commits": total_commits,
                "modified": modified,
                "untracked": untracked,
                "status": status
            }
        except Exception as e:
            return f"Error retrieving Git info: {e}"

    def get_project_stats(self):
        py_files = list(self.root_dir.rglob("*.py"))
        all_files = [f for f in self.root_dir.rglob("*") if f.is_file() and ".git" not in f.parts and "__pycache__" not in f.parts]
        folders = [d for d in self.root_dir.rglob("*") if d.is_dir() and ".git" not in d.parts and "__pycache__" not in d.parts]
        
        total_loc = 0
        file_lengths = {}
        
        for pf in py_files:
            try:
                with open(pf, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_loc += len(lines)
                    file_lengths[pf.name] = len(lines)
            except:
                continue
                
        if not file_lengths:
            return {
                "folders": len(folders), "py_files": len(py_files), "all_files": len(all_files),
                "loc": 0, "largest": "N/A", "smallest": "N/A", "avg": 0
            }

        largest = max(file_lengths, key=file_lengths.get)
        smallest = min(file_lengths, key=file_lengths.get)
        avg = total_loc / len(py_files) if py_files else 0
        
        return {
            "folders": len(folders),
            "py_files": len(py_files),
            "all_files": len(all_files),
            "loc": total_loc,
            "largest": f"{largest} ({file_lengths[largest]} lines)",
            "smallest": f"{smallest} ({file_lengths[smallest]} lines)",
            "avg": round(avg, 2)
        }

    def generate_tree(self, path, prefix=""):
        tree_str = ""
        space = '    '
        branch = '│   '
        tee = '├── '
        last = '└── '
        
        contents = sorted([p for p in path.iterdir() if p.name not in [".git", "__pycache__", ".ipynb_checkpoints", ".pytest_cache"]])
        pointers = [tee] * (len(contents) - 1) + [last]
        
        for pointer, p in zip(pointers, contents):
            tree_str += f"{prefix}{pointer}{p.name}\n"
            if p.is_dir():
                extension = branch if pointer == tee else space
                tree_str += self.generate_tree(p, prefix=prefix + extension)
        return tree_str

    def get_env_info(self):
        info = {
            "python": sys.version,
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor(),
            "ram": f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB",
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        }
        if torch:
            info["torch"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
        else:
            info["torch"] = "Not Installed"
            info["cuda_available"] = False
            info["cuda_version"] = "N/A"
        return info

    def get_progress_bar(self, percentage):
        filled = int(percentage / 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"{bar} {percentage}%"

    def get_embedded_images(self):
        img_markdown = ""
        valid_ext = [".jpg", ".jpeg", ".png", ".webp"]
        found_images = []
        
        for d in self.output_dirs:
            path = self.root_dir / d
            if path.exists():
                for img_path in path.rglob("*"):
                    if img_path.suffix.lower() in valid_ext:
                        found_images.append(img_path)
        
        for img in found_images[-5:]: # Last 5 images
            try:
                with open(img, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode()
                img_markdown += f"### {img.name}\n![{img.name}](data:image/{img.suffix[1:]};base64,{encoded})\n\n"
            except:
                continue
        return img_markdown if img_markdown else "No recent screenshots found."

    def get_video_links(self):
        vid_md = ""
        if self.video_dir.exists():
            vids = list(self.video_dir.glob("*.mp4"))
            if vids:
                vid_md = "| Video Name | Path |\n|---|---|\n"
                for v in vids:
                    vid_md += f"| {v.name} | `{v.relative_to(self.root_dir)}` |\n"
        return vid_md if vid_md else "No videos found in outputs/videos/."

    def run_commands(self):
        try:
            py_ver = subprocess.check_output(["python", "--version"], stderr=subprocess.STDOUT).decode().strip()
            git_st = subprocess.check_output(["git", "status", "--short"], stderr=subprocess.STDOUT).decode().strip()
            return py_ver, git_st
        except:
            return "N/A", "N/A"

    def create_new_entry(self):
        ts = self.get_timestamp()
        git_info = self.get_git_info()
        stats = self.get_project_stats()
        env = self.get_env_info()
        py_v, git_v = self.run_commands()
        
        # Progress estimates (Mock values logic - can be automated via config files later)
        progress = {
            "Detection": 85,
            "Tracking": 70,
            "Pose Estimation": 60,
            "Motion Recovery": 40,
            "Visualization": 50,
            "Trajectory Alignment": 30,
            "Export": 20
        }

        cells = []
        
        # Section 1: Header
        cells.append(new_markdown_cell(f"""# 📝 Progress Log: {ts['date']}
**Student Name:** Paggilla Saketh  
**University:** Amrita Vishwa Vidyapeetham  
**Time:** {ts['time']}  
**Day:** {ts['day']}  
---"""))

        # Section 2: Project Overview
        cells.append(new_markdown_cell(f"""## 🎯 Project Overview
- **Current Objective:** Integrating multi-person tracking with 3D pose estimation.
- **Current Milestone:** Milestone 3 - Temporal Consistency in Motion Recovery.
- **Status:** 🟢 In Progress"""))

        # Section 3: Git Information
        git_md = "## 🌿 Git Information\n"
        if isinstance(git_info, dict):
            git_md += f"- **Branch:** `{git_info['branch']}`\n"
            git_md += f"- **Latest Commit:** `{git_info['hash'][:8]}`\n"
            git_md += f"- **Message:** {git_info['message']}\n"
            git_md += f"- **Author:** {git_info['author']}\n"
            git_md += f"- **Date:** {git_info['date']}\n"
            git_md += f"- **Total Commits:** {git_info['total_commits']}\n"
            git_md += f"\n<details><summary><b>Git Status Detail</b></summary>\n\n```\n{git_info['status']}\n```\n</details>"
        else:
            git_md += git_info
        cells.append(new_markdown_cell(git_md))

        # Section 4: Project Statistics
        cells.append(new_markdown_cell(f"""## 📊 Project Statistics
| Metric | Value |
| :--- | :--- |
| **Total Folders** | {stats['folders']} |
| **Python Files** | {stats['py_files']} |
| **Total Files** | {stats['all_files']} |
| **Total Lines (Python)** | {stats['loc']} |
| **Largest File** | `{stats['largest']}` |
| **Smallest File** | `{stats['smallest']}` |
| **Avg File Length** | {stats['avg']} lines |"""))

        # Section 5: Folder Structure
        cells.append(new_markdown_cell(f"## 📂 Project Folder Structure\n```\n{self.root_dir.name}/\n{self.generate_tree(self.root_dir)}\n```"))

        # Section 6: Today's Modified Files
        mod_md = "## 🛠 Today's Modified Files\n"
        if isinstance(git_info, dict) and git_info['modified']:
            mod_md += "| Filename | Last Modified |\n| :--- | :--- |\n"
            for f in git_info['modified']:
                f_path = self.root_dir / f
                mtime = datetime.fromtimestamp(f_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S') if f_path.exists() else "Deleted"
                mod_md += f"| `{f}` | {mtime} |\n"
        else:
            mod_md += "No files modified since last commit."
        cells.append(new_markdown_cell(mod_md))

        # Section 7: Project Progress
        prog_md = "## 📈 Project Progress\n"
        for k, v in progress.items():
            prog_md += f"**{k}**\n`{self.get_progress_bar(v)}`\n\n"
        cells.append(new_markdown_cell(prog_md))

        # Section 8: Environment
        cells.append(new_markdown_cell(f"""## 💻 Environment
- **Python:** `{env['python'].split()[0]}`
- **OS:** `{env['os']}`
- **CPU:** `{env['cpu']}`
- **RAM:** `{env['ram']}`
- **OpenCV:** `{env['opencv']}`
- **NumPy:** `{env['numpy']}`
- **Torch:** `{env['torch']}`
- **CUDA:** `{'✅ Available' if env['cuda_available'] else '❌ Not Available'}` (Version: {env['cuda_version']})"""))

        # Section 9: Execution Log
        cells.append(new_markdown_cell(f"## 📜 Execution Log\n**Python Version Command:**\n```\n{py_v}\n```\n**Git Status Command:**\n```\n{git_v}\n```"))

        # Section 10-13: Notes & Work
        msg = git_info['message'] if isinstance(git_info, dict) else "N/A"
        cells.append(new_markdown_cell(f"""## 📝 Research Notes
*Add research insights here...*

## 🔨 Today's Work ({ts['date']})
- **Time:** {ts['time']}
- **Trigger Commit:** {msg}
- Updated project structure and automated logging.

## ⚠️ Issues Found
1. [ ] 
2. [ ] 

## ✅ Solutions Implemented
1. Automated notebook generation.
"""))

        # Section 14-15: Multimedia
        cells.append(new_markdown_cell(f"## 🖼 Screenshots\n{self.get_embedded_images()}"))
        cells.append(new_markdown_cell(f"## 🎥 Output Videos\n{self.get_video_links()}"))

        # Section 16-18: Performance & Admin
        cells.append(new_markdown_cell(f"""## 🚀 Performance
| Metric | Value |
| :--- | :--- |
| **Current FPS** | 0.0 |
| **Latency** | 0.0ms |
| **Tracking Accuracy** | N/A |

## 📅 Next Tasks
- [ ] Implement Kalman Filter for smoothing.
- [ ] Expand dataset for validation.
- [ ] Prepare midterm presentation.

## 👨‍🏫 Supervisor Comments
> """))

        # Section 20: Summary
        cells.append(new_markdown_cell(f"## 🏁 Summary\nCompleted execution of `update_notebook.py` at {ts['time']}. Project metrics and Git status synchronized."))

        return cells

    def update_notebook(self):
        new_cells = self.create_new_entry()
        
        if self.notebook_path.exists():
            with open(self.notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Maintain previous work by appending. 
            # We add a separator.
            separator = new_markdown_cell("--- \n # PREVIOUS LOGS \n ---")
            
            # Check if separator already exists to avoid duplication
            has_sep = any("PREVIOUS LOGS" in c.source for c in nb.cells if c.cell_type == 'markdown')
            
            if not has_sep:
                nb.cells = new_cells + [separator] + nb.cells
            else:
                # Add newest content at the top
                nb.cells = new_cells + nb.cells
        else:
            nb = new_notebook()
            nb.metadata.kernelspec = {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
            nb.cells = new_cells

        with open(self.notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"✅ Notebook updated successfully at: {self.notebook_path}")

if __name__ == "__main__":
    tracker = ProjectTracker()
    tracker.update_notebook()