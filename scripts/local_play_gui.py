#!/usr/bin/env python3
r"""Tkinter GUI for launching local MuJoCo policy playback.

Run from PowerShell:
    cd D:\Desktop_Files\GPU-Train\RTX6000\Magicbot_Z1
    python .\scripts\local_play_gui.py
"""

from __future__ import annotations

import queue
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MJCF_DEFAULT = PROJECT_ROOT / "magicbot-z1_description" / "mjcf" / "MAGICBOTZ1.xml"
GUI_CONTROL_FILE = Path(tempfile.gettempdir()) / "magicbot_z1_local_play_control.txt"
EXPORT_JIT_SCRIPT = PROJECT_ROOT / "magiclab_rl_lab" / "scripts" / "export_jit.py"

GUI_KEY_COMMANDS = {
    "Up": ("up", "Forward +0.1"),
    "Down": ("down", "Backward -0.1"),
    "Left": ("left", "Yaw left +0.1"),
    "Right": ("right", "Yaw right -0.1"),
    "q": ("q", "Lateral left +0.1"),
    "e": ("e", "Lateral right -0.1"),
    "space": ("space", "Stop"),
    "Escape": ("esc", "Quit"),
}

CONTROL_LABELS = {
    "up": "Forward +0.1",
    "down": "Backward -0.1",
    "left": "Yaw left +0.1",
    "right": "Yaw right -0.1",
    "q": "Lateral left +0.1",
    "e": "Lateral right -0.1",
    "space": "Stop",
    "esc": "Quit",
}

VEL_LINE_RE = re.compile(r"\[CMD\]\s+vel=\(([-+0-9.]+),\s*([-+0-9.]+),\s*([-+0-9.]+)\)")


PRESET_RUNTIME_BINDINGS: dict[str, dict[str, object]] = {
    "p1_coarse": {"phase": "p1_coarse", "terrain": "", "flat": False},
    "p1_fine": {"phase": "p1_fine", "terrain": "", "flat": False},
    "p2_coarse": {"phase": "p2_coarse", "terrain": "", "flat": False},
    "p2_fine": {"phase": "p2_fine", "terrain": "", "flat": False},
    "p3_coarse": {"phase": "p3_coarse", "terrain": "p3_coarse", "flat": False},
    "p3_fine": {"phase": "p3_fine", "terrain": "p3_fine", "flat": False},
}


def runtime_binding_from_preset(name: str) -> dict[str, object]:
    binding = PRESET_RUNTIME_BINDINGS.get(name)
    if binding is not None:
        return dict(binding)
    return {"phase": "", "terrain": "", "flat": False}


def phase_family(name: str) -> str:
    return name.split("_", 1)[0] if "_" in name else name


def find_deploy_cfg_for_phase(phase_name: str) -> Path | None:
    """Find the best available deploy.yaml for a preset.

    Priority:
    1. Exact phase artifact under videos/p/<phase>/params
    2. Exact phase artifact nested anywhere below videos/p/<phase>
    3. Exact phase artifact under models/p/<phase>/params
    4. Same phase family sibling (prefer coarse before fine) exact params
    5. Same phase family sibling nested deploy
    6. No deploy config; caller may omit --deploy_cfg because mujoco_manual.py
       can run with its built-in defaults.
    """
    video_root = PROJECT_ROOT / "videos" / "p"
    models_root = PROJECT_ROOT / "models" / "p"

    def nested_latest(root: Path) -> Path | None:
        if not root.exists():
            return None
        candidates = sorted(
            root.rglob("deploy.yaml"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    direct_candidates = [
        video_root / phase_name / "params" / "deploy.yaml",
        models_root / phase_name / "params" / "deploy.yaml",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    exact_nested = nested_latest(video_root / phase_name)
    if exact_nested is not None:
        return exact_nested

    family = phase_family(phase_name)
    sibling_names: list[str] = []
    if phase_name.endswith("_fine"):
        sibling_names.append(f"{family}_coarse")
    elif phase_name.endswith("_coarse"):
        sibling_names.append(f"{family}_fine")

    sibling_names.extend(
        sorted(
            {
                path.name
                for path in models_root.iterdir()
                if path.is_dir() and phase_family(path.name) == family and path.name != phase_name
            }
        )
    )

    seen: set[str] = set()
    ordered_siblings = [name for name in sibling_names if not (name in seen or seen.add(name))]

    for sibling in ordered_siblings:
        for candidate in (
            video_root / sibling / "params" / "deploy.yaml",
            models_root / sibling / "params" / "deploy.yaml",
        ):
            if candidate.exists():
                return candidate
        sibling_nested = nested_latest(video_root / sibling)
        if sibling_nested is not None:
            return sibling_nested

    return None


def rel_display(path: str | Path) -> str:
    path = Path(path)
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
        return ".\\" + str(rel).replace("/", "\\")
    except Exception:
        return str(path)


def sort_policy_candidates(paths: list[Path]) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        priority = 0 if "policy" in name else 1
        return (priority, name)

    return sorted(paths, key=sort_key)


def classify_policy_file(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "policy" in name and "model_" not in name:
        return "jit"
    return "checkpoint"


def format_policy_label(path: str | Path) -> str:
    policy_path = Path(path)
    kind = classify_policy_file(policy_path)
    prefix = "JIT" if kind == "jit" else "CKPT"
    return f"[{prefix}] {policy_path.name}"


def scan_presets() -> dict[str, dict[str, object]]:
    presets: dict[str, dict[str, object]] = {}
    models_root = PROJECT_ROOT / "models" / "p"
    video_root = PROJECT_ROOT / "videos" / "p"

    if not models_root.exists():
        return presets

    for phase_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        policy_files = sort_policy_candidates(list(phase_dir.glob("*.pt")))
        if not policy_files:
            continue

        deploy_cfg = find_deploy_cfg_for_phase(phase_dir.name)
        runtime_binding = runtime_binding_from_preset(phase_dir.name)

        presets[phase_dir.name] = {
            "mjcf": str(MJCF_DEFAULT),
            "policy": str(policy_files[0]),
            "policy_files": [str(path) for path in policy_files],
            "deploy_cfg": str(deploy_cfg) if deploy_cfg is not None else "",
            "phase": str(runtime_binding["phase"]),
            "terrain": str(runtime_binding["terrain"]),
            "flat": "1" if bool(runtime_binding["flat"]) else "0",
        }

    return presets


class LocalPlayGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MagicBot Z1 local_play GUI")
        self.configure_window_geometry()

        self.presets = scan_presets()
        self.proc: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.preset_var = tk.StringVar(value=next(iter(self.presets), "p2_coarse"))
        self.mjcf_var = tk.StringVar(value=rel_display(MJCF_DEFAULT))
        self.policy_var = tk.StringVar()
        self.policy_choice_var = tk.StringVar()
        self.deploy_var = tk.StringVar()
        self.phase_var = tk.StringVar(value="p2")
        self.terrain_var = tk.StringVar(value="")
        self.csv_path_var = tk.StringVar(value="")
        self.steps_var = tk.StringVar(value="10000")
        self.vel_x_var = tk.StringVar(value="0.3")
        self.vel_y_var = tk.StringVar(value="0.0")
        self.vel_yaw_var = tk.StringVar(value="0.0")
        self.csv_var = tk.BooleanVar(value=False)
        self.flat_var = tk.BooleanVar(value=False)
        self.random_spawn_var = tk.BooleanVar(value=False)
        self.key_status_var = tk.StringVar(value="GUI keyboard idle. Click the right panel to capture keys.")
        self.live_vel_var = tk.StringVar(value="cmd vel = (0.30, 0.00, 0.00)")

        self.command_preview = tk.Text(root, height=8, wrap="word")
        self.output_text = tk.Text(root, height=20, wrap="word", bg="#0f172a", fg="#e5e7eb")
        self.status_var = tk.StringVar(value=f"Project root: {PROJECT_ROOT}")
        self.policy_choice_buttons: list[ttk.Radiobutton] = []

        self._build()
        self.apply_preset()
        self.root.after(120, self.flush_log_queue)

    def configure_window_geometry(self) -> None:
        screen_w = max(self.root.winfo_screenwidth(), 1280)
        screen_h = max(self.root.winfo_screenheight(), 720)
        width = min(980, max(760, int(screen_w * 0.40)))
        height = min(860, max(620, int(screen_h * 0.82)))
        x = 16
        y = max(12, int(screen_h * 0.04))
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(720, 580)
        self.root.resizable(True, True)

    def _build(self) -> None:
        shell = ttk.Frame(self.root)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        self.scroll_canvas = tk.Canvas(shell, highlightthickness=0)
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")
        canvas_scroll = ttk.Scrollbar(shell, orient="vertical", command=self.scroll_canvas.yview)
        canvas_scroll.grid(row=0, column=1, sticky="ns")
        self.scroll_canvas.configure(yscrollcommand=canvas_scroll.set)

        outer = ttk.Frame(self.scroll_canvas, padding=14)
        self.scroll_window = self.scroll_canvas.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", self.on_outer_configure)
        self.scroll_canvas.bind("<Configure>", self.on_canvas_configure)
        self.scroll_canvas.bind_all("<MouseWheel>", self.on_mousewheel)

        top = ttk.Frame(outer)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, text="Preset").grid(row=0, column=0, sticky="w")
        preset_values = list(self.presets) or ["p2_coarse"]
        preset_box = ttk.Combobox(top, textvariable=self.preset_var, values=preset_values, state="readonly", width=20)
        preset_box.grid(row=0, column=1, sticky="w", padx=(8, 8))
        preset_box.bind("<<ComboboxSelected>>", lambda _event: self.apply_preset())
        ttk.Button(top, text="Apply Preset", command=self.apply_preset).grid(row=0, column=2, sticky="w")
        ttk.Button(top, text="Run Policy", command=self.run_policy).grid(row=0, column=3, sticky="w", padx=(12, 0))
        ttk.Button(top, text="Stop", command=self.stop_policy).grid(row=0, column=4, sticky="w", padx=(8, 0))
        ttk.Button(top, text="Copy Cmd", command=self.copy_command).grid(row=0, column=5, sticky="w", padx=(8, 0))

        main_pane = ttk.Panedwindow(outer, orient="horizontal")
        main_pane.pack(fill="both", pady=(0, 10), expand=False)

        form = ttk.LabelFrame(main_pane, text="Inputs", padding=12)

        self._path_row(form, 0, "MJCF", self.mjcf_var, ("XML files", "*.xml"))
        self._path_row(form, 1, "Policy", self.policy_var, ("PyTorch files", "*.pt"))
        self._path_row(form, 2, "Deploy cfg", self.deploy_var, ("YAML files", "*.yaml"))

        ttk.Label(form, text="Preset policies").grid(row=3, column=0, sticky="nw", pady=6)
        self.policy_choice_frame = ttk.Frame(form)
        self.policy_choice_frame.grid(row=3, column=1, columnspan=5, sticky="ew", padx=(8, 0), pady=6)

        ttk.Label(form, text="Phase").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Entry(form, textvariable=self.phase_var, width=18).grid(row=4, column=1, sticky="w", padx=(8, 20))
        ttk.Label(form, text="Terrain").grid(row=4, column=2, sticky="w")
        ttk.Entry(form, textvariable=self.terrain_var, width=18).grid(row=4, column=3, sticky="w", padx=(8, 20))
        ttk.Label(form, text="Num steps").grid(row=4, column=4, sticky="w")
        ttk.Entry(form, textvariable=self.steps_var, width=12).grid(row=4, column=5, sticky="w", padx=(8, 0))

        ttk.Label(form, text="vel_x").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Entry(form, textvariable=self.vel_x_var, width=12).grid(row=5, column=1, sticky="w", padx=(8, 20))
        ttk.Label(form, text="vel_y").grid(row=5, column=2, sticky="w")
        ttk.Entry(form, textvariable=self.vel_y_var, width=12).grid(row=5, column=3, sticky="w", padx=(8, 20))
        ttk.Label(form, text="vel_yaw").grid(row=5, column=4, sticky="w")
        ttk.Entry(form, textvariable=self.vel_yaw_var, width=12).grid(row=5, column=5, sticky="w", padx=(8, 0))

        options = ttk.Frame(form)
        options.grid(row=6, column=0, columnspan=6, sticky="w", pady=(8, 0))
        ttk.Label(options, text="keyboard: always on").pack(side="left", padx=(0, 12))
        ttk.Checkbutton(options, text="csv", variable=self.csv_var, command=self.refresh_preview).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(options, text="flat", variable=self.flat_var, command=self.refresh_preview).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(options, text="random spawn", variable=self.random_spawn_var, command=self.refresh_preview).pack(side="left", padx=(0, 12))

        ttk.Label(form, text="CSV path (optional)").grid(row=7, column=0, sticky="w", pady=6)
        ttk.Entry(form, textvariable=self.csv_path_var, width=72).grid(row=7, column=1, columnspan=4, sticky="ew", padx=(8, 8))
        ttk.Button(form, text="Browse", command=self.browse_csv_path).grid(row=7, column=5, sticky="w")

        keyboard_frame = ttk.LabelFrame(main_pane, text="Keyboard Control", padding=12)
        ttk.Label(
            keyboard_frame,
            text="Use GUI to control running MuJoCo.\nClick below, then press Arrow / Q / E / Space / Esc.",
            justify="left",
        ).pack(anchor="w")
        ttk.Button(keyboard_frame, text="Focus Keyboard Here", command=self.focus_keyboard_capture).pack(anchor="w", pady=(8, 8))
        ttk.Label(keyboard_frame, textvariable=self.key_status_var, justify="left", foreground="#1d4ed8").pack(anchor="w", pady=(0, 8))
        ttk.Label(keyboard_frame, textvariable=self.live_vel_var, justify="left", foreground="#047857").pack(anchor="w", pady=(0, 8))

        button_grid = ttk.Frame(keyboard_frame)
        button_grid.pack(anchor="w", pady=(0, 10))
        ttk.Button(button_grid, text="Forward", width=12, command=lambda: self.send_control_command("up")).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(button_grid, text="Left Turn", width=12, command=lambda: self.send_control_command("left")).grid(row=1, column=0, padx=4, pady=4)
        ttk.Button(button_grid, text="Stop", width=12, command=lambda: self.send_control_command("space")).grid(row=1, column=1, padx=4, pady=4)
        ttk.Button(button_grid, text="Right Turn", width=12, command=lambda: self.send_control_command("right")).grid(row=1, column=2, padx=4, pady=4)
        ttk.Button(button_grid, text="Backward", width=12, command=lambda: self.send_control_command("down")).grid(row=2, column=1, padx=4, pady=4)
        ttk.Button(button_grid, text="Shift Left", width=12, command=lambda: self.send_control_command("q")).grid(row=3, column=0, padx=4, pady=4)
        ttk.Button(button_grid, text="Shift Right", width=12, command=lambda: self.send_control_command("e")).grid(row=3, column=2, padx=4, pady=4)
        ttk.Button(button_grid, text="Quit", width=12, command=lambda: self.send_control_command("esc")).grid(row=3, column=1, padx=4, pady=4)

        self.keyboard_capture = tk.Text(
            keyboard_frame,
            width=30,
            height=8,
            wrap="word",
            relief="solid",
            borderwidth=1,
            bg="#f8fafc",
            fg="#0f172a",
        )
        keyboard_text_frame = ttk.Frame(keyboard_frame)
        keyboard_text_frame.pack(fill="both", expand=True)
        self.keyboard_capture.pack(in_=keyboard_text_frame, side="left", fill="both", expand=True)
        keyboard_scroll = ttk.Scrollbar(keyboard_text_frame, orient="vertical", command=self.keyboard_capture.yview)
        keyboard_scroll.pack(side="right", fill="y")
        self.keyboard_capture.configure(yscrollcommand=keyboard_scroll.set)
        self.keyboard_capture.insert(
            "1.0",
            "GUI key map\n"
            "Up    -> Forward +0.1\n"
            "Down  -> Backward -0.1\n"
            "Left  -> Yaw left +0.1\n"
            "Right -> Yaw right -0.1\n"
            "Q/E   -> Lateral +/-0.1\n"
            "Space -> Stop\n"
            "Esc   -> Quit\n\n"
            "Recommended:\n"
            "1. Run Policy\n"
            "2. Click this panel\n"
            "3. Press keys here\n"
        )
        self.keyboard_capture.bind("<KeyPress>", self.handle_capture_keypress)
        self.keyboard_capture.bind("<Button-1>", lambda _event: self.focus_keyboard_capture())
        self.keyboard_capture.bind("<FocusIn>", lambda _event: self.key_status_var.set("GUI keyboard armed. Keys go to MuJoCo process."))

        main_pane.add(form, weight=5)
        main_pane.add(keyboard_frame, weight=3)

        content_pane = ttk.Panedwindow(outer, orient="vertical")
        content_pane.pack(fill="both", pady=(0, 10), expand=True)

        preview_frame = ttk.LabelFrame(content_pane, text="Command Preview", padding=12)
        preview_body = ttk.Frame(preview_frame)
        preview_body.pack(fill="both", expand=True)
        self.command_preview.pack(in_=preview_body, side="left", fill="both", expand=True)
        preview_scroll = ttk.Scrollbar(preview_body, orient="vertical", command=self.command_preview.yview)
        preview_scroll.pack(side="right", fill="y")
        self.command_preview.configure(yscrollcommand=preview_scroll.set)
        self.command_preview.configure(state="disabled")

        output_frame = ttk.LabelFrame(content_pane, text="Process Output", padding=12)
        output_body = ttk.Frame(output_frame)
        output_body.pack(fill="both", expand=True)
        self.output_text.pack(in_=output_body, side="left", fill="both", expand=True)
        output_scroll = ttk.Scrollbar(output_body, orient="vertical", command=self.output_text.yview)
        output_scroll.pack(side="right", fill="y")
        self.output_text.configure(yscrollcommand=output_scroll.set)
        self.output_text.configure(state="disabled")

        content_pane.add(preview_frame, weight=1)
        content_pane.add(output_frame, weight=3)

        status = ttk.Label(outer, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", pady=(8, 0))

        watchers = [
            self.mjcf_var,
            self.policy_var,
            self.deploy_var,
            self.phase_var,
            self.terrain_var,
            self.csv_path_var,
            self.steps_var,
            self.vel_x_var,
            self.vel_y_var,
            self.vel_yaw_var,
        ]
        for var in watchers:
            var.trace_add("write", lambda *_: self.refresh_preview())

    def on_outer_configure(self, _event: tk.Event) -> None:
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def on_canvas_configure(self, event: tk.Event) -> None:
        self.scroll_canvas.itemconfigure(self.scroll_window, width=event.width)

    def on_mousewheel(self, event: tk.Event) -> str | None:
        widget = event.widget
        if widget in (self.command_preview, self.output_text, self.keyboard_capture):
            return None
        delta = int(-event.delta / 120) if event.delta else 0
        if delta:
            self.scroll_canvas.yview_scroll(delta, "units")
            return "break"
        return None

    def focus_keyboard_capture(self) -> None:
        self.keyboard_capture.focus_set()
        self.key_status_var.set("GUI keyboard armed. Keys go to MuJoCo process.")

    def _is_torchscript_policy(self, path: Path) -> bool:
        try:
            import torch

            torch.jit.load(str(path), map_location="cpu")
            return True
        except Exception:
            return False

    def prepare_policy_for_launch(self) -> Path | None:
        selected = self.resolve_input_path(self.policy_var.get())
        if not selected.exists():
            messagebox.showerror("Missing policy", f"Policy file does not exist:\n{selected}")
            return None

        if self._is_torchscript_policy(selected):
            self.status_var.set(f"Using JIT policy: {selected.name}")
            return selected

        if not EXPORT_JIT_SCRIPT.exists():
            messagebox.showerror("Missing exporter", f"Export script not found:\n{EXPORT_JIT_SCRIPT}")
            return None

        export_dir = selected.parent / "gui_exported"
        export_path = export_dir / f"{selected.stem}_policy.pt"
        needs_export = (not export_path.exists()) or (export_path.stat().st_mtime < selected.stat().st_mtime)

        if not needs_export and self._is_torchscript_policy(export_path):
            self.status_var.set(f"Using cached exported policy: {export_path.name}")
            return export_path

        self.append_output(
            f"[EXPORT] checkpoint -> JIT\n"
            f"  checkpoint: {selected}\n"
            f"  output: {export_path}\n"
        )
        self.status_var.set(f"Exporting checkpoint: {selected.name}")
        self.root.update_idletasks()

        command = [
            sys.executable,
            str(EXPORT_JIT_SCRIPT),
            "--checkpoint",
            str(selected),
            "--output",
            str(export_path),
        ]
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.append_output(result.stdout)
        if result.returncode != 0:
            messagebox.showerror(
                "Export failed",
                f"Could not export checkpoint as JIT policy.\n\n{selected.name}\n\nSee Process Output for details.",
            )
            self.status_var.set(f"Export failed: {selected.name}")
            return None

        self.policy_var.set(rel_display(export_path))
        self.policy_choice_var.set(str(selected.resolve()))
        self.status_var.set(f"Exported checkpoint and launching: {selected.name}")
        return export_path

    def render_policy_buttons(self, preset_name: str) -> None:
        for button in self.policy_choice_buttons:
            button.destroy()
        self.policy_choice_buttons.clear()

        preset = self.presets.get(preset_name)
        if not preset:
            return

        policy_files = [Path(path) for path in preset.get("policy_files", [])]
        current_policy = self.policy_var.get().strip()
        if current_policy:
            self.policy_choice_var.set(str(Path(current_policy).resolve()))
        elif policy_files:
            self.policy_choice_var.set(str(policy_files[0].resolve()))

        for idx, policy_path in enumerate(policy_files):
            label = format_policy_label(policy_path)
            button = ttk.Radiobutton(
                self.policy_choice_frame,
                text=label,
                value=str(policy_path.resolve()),
                variable=self.policy_choice_var,
                command=self.on_policy_choice_changed,
            )
            row = idx // 3
            col = idx % 3
            button.grid(row=row, column=col, sticky="w", padx=(0, 10), pady=(0, 4))
            self.policy_choice_buttons.append(button)

    def on_policy_choice_changed(self) -> None:
        choice = self.policy_choice_var.get().strip()
        if not choice:
            return
        self.policy_var.set(rel_display(choice))
        kind = classify_policy_file(choice)
        mode = "JIT policy" if kind == "jit" else "checkpoint (auto-export on run)"
        self.status_var.set(f"Selected {mode}: {Path(choice).name}")
        self.refresh_preview()

    def _append_control_command(self, command: str) -> None:
        GUI_CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with GUI_CONTROL_FILE.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.time():.6f}\t{command}\n")

    def update_live_velocity(self, vx: float | str, vy: float | str, vyaw: float | str) -> None:
        self.live_vel_var.set(f"cmd vel = ({float(vx):.2f}, {float(vy):.2f}, {float(vyaw):.2f})")

    def send_control_command(self, command: str) -> None:
        if not self.proc or self.proc.poll() is not None:
            self.key_status_var.set("No running local_play process.")
            return
        self._append_control_command(command)
        self.key_status_var.set(f"Sent: {CONTROL_LABELS.get(command, command)}")
        self.append_output(f"[GUI CTRL] {CONTROL_LABELS.get(command, command)}\n")
        self.focus_keyboard_capture()

    def handle_capture_keypress(self, event: tk.Event) -> str:
        key = event.keysym
        if key not in GUI_KEY_COMMANDS:
            self.key_status_var.set(f"GUI keyboard ignores key: {key}")
            return "break"

        command, label = GUI_KEY_COMMANDS[key]
        self.send_control_command(command)
        self.key_status_var.set(f"Sent: {label}")
        return "break"

    def _path_row(self, parent: ttk.LabelFrame, row: int, label: str, var: tk.StringVar, filetypes: tuple[str, str]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=var, width=72).grid(row=row, column=1, columnspan=4, sticky="ew", padx=(8, 8))
        ttk.Button(parent, text="Browse", command=lambda: self.browse_file(var, filetypes)).grid(row=row, column=5, sticky="w")

    def browse_file(self, var: tk.StringVar, filetype: tuple[str, str]) -> None:
        start_dir = PROJECT_ROOT
        try:
            current = Path(var.get())
            if current.exists():
                start_dir = current.parent
        except Exception:
            pass

        path = filedialog.askopenfilename(
            parent=self.root,
            initialdir=start_dir,
            filetypes=[filetype, ("All files", "*.*")],
        )
        if path:
            var.set(rel_display(path))

    def browse_csv_path(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            initialdir=PROJECT_ROOT / "logs",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path_var.set(rel_display(path))
            self.csv_var.set(True)
            self.refresh_preview()

    def apply_preset(self) -> None:
        preset = self.presets.get(self.preset_var.get())
        if not preset:
            self.refresh_preview()
            return

        self.mjcf_var.set(rel_display(preset["mjcf"]))
        default_policy = str(preset["policy"])
        self.policy_choice_var.set(str(Path(default_policy).resolve()))
        self.policy_var.set(rel_display(default_policy))
        deploy_cfg = preset.get("deploy_cfg", "")
        self.deploy_var.set(rel_display(deploy_cfg) if deploy_cfg else "")
        self.phase_var.set(preset["phase"])
        self.terrain_var.set(preset.get("terrain", ""))
        self.csv_path_var.set("")
        self.csv_var.set(False)
        self.flat_var.set(preset.get("flat", "0") == "1")
        self.random_spawn_var.set(False)
        self.update_live_velocity(self.vel_x_var.get(), self.vel_y_var.get(), self.vel_yaw_var.get())
        phase = preset.get("phase", "")
        terrain = preset.get("terrain", "")
        terrain_label = terrain or ("flat-from-artifact" if phase else "manual/none")
        deploy_label = Path(deploy_cfg).name if deploy_cfg else "optional / default PD"
        self.status_var.set(
            f"Preset {self.preset_var.get()} loaded: phase={phase or '-'}, terrain={terrain_label}, deploy={deploy_label}"
        )
        self.render_policy_buttons(self.preset_var.get())
        self.refresh_preview()

    def resolve_input_path(self, text: str) -> Path:
        if not text.strip():
            return Path()
        text = text.strip()
        if text.startswith(".\\") or text.startswith("./"):
            return (PROJECT_ROOT / text[2:]).resolve()
        return Path(text).expanduser().resolve()

    def build_args(self) -> list[str]:
        return self.build_args_with_policy(self.resolve_input_path(self.policy_var.get()))

    def build_args_with_policy(self, policy_path: Path) -> list[str]:
        args = [
            str(PROJECT_ROOT / "sim2sim" / "mujoco_manual.py"),
            "--mjcf", str(self.resolve_input_path(self.mjcf_var.get())),
            "--policy", str(policy_path),
            "--control_file", str(GUI_CONTROL_FILE),
        ]
        deploy_cfg = self.resolve_input_path(self.deploy_var.get())
        if str(deploy_cfg):
            args += ["--deploy_cfg", str(deploy_cfg)]

        phase = self.phase_var.get().strip()
        terrain = self.terrain_var.get().strip()
        steps = self.steps_var.get().strip()
        vel_x = self.vel_x_var.get().strip()
        vel_y = self.vel_y_var.get().strip()
        vel_yaw = self.vel_yaw_var.get().strip()

        if phase:
            args += ["--phase", phase]
        if terrain:
            args += ["--terrain", terrain]
        if steps:
            args += ["--num_steps", steps]
        if vel_x:
            args += ["--vel_x", vel_x]
        if vel_y:
            args += ["--vel_y", vel_y]
        if vel_yaw:
            args += ["--vel_yaw", vel_yaw]
        args.append("--keyboard")
        if self.flat_var.get():
            args.append("--flat")
        if self.random_spawn_var.get():
            args.append("--random_spawn")
        if self.csv_var.get():
            csv_path = self.csv_path_var.get().strip()
            args.append("--csv")
            if csv_path:
                args.append(str(self.resolve_input_path(csv_path)))

        return args

    def render_preview(self) -> str:
        args = self.build_args()
        lines = [f"{shlex.quote(sys.executable)} `"]
        for index, arg in enumerate(args):
            suffix = " `" if index < len(args) - 1 else ""
            lines.append(f"  {arg}{suffix}")
        return "\n".join(lines)

    def refresh_preview(self) -> None:
        try:
            self.update_live_velocity(self.vel_x_var.get(), self.vel_y_var.get(), self.vel_yaw_var.get())
        except ValueError:
            self.live_vel_var.set("cmd vel = (invalid input)")
        preview = self.render_preview()
        self.command_preview.configure(state="normal")
        self.command_preview.delete("1.0", "end")
        self.command_preview.insert("1.0", preview)
        self.command_preview.configure(state="disabled")

    def validate(self) -> bool:
        required = {
            "MJCF": self.resolve_input_path(self.mjcf_var.get()),
            "Policy": self.resolve_input_path(self.policy_var.get()),
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            messagebox.showerror("Missing files", f"These paths do not exist:\n- " + "\n- ".join(missing))
            return False
        deploy_path = self.resolve_input_path(self.deploy_var.get())
        if str(deploy_path) and not deploy_path.exists():
            messagebox.showerror("Missing files", f"Deploy cfg path does not exist:\n- {deploy_path}")
            return False
        return True

    def append_output(self, text: str) -> None:
        match = VEL_LINE_RE.search(text)
        if match:
            self.update_live_velocity(match.group(1), match.group(2), match.group(3))
        self.output_text.configure(state="normal")
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.output_text.configure(state="disabled")

    def run_policy(self) -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Running", "A local_play process is already running.")
            return
        if not self.validate():
            return

        runnable_policy = self.prepare_policy_for_launch()
        if runnable_policy is None:
            return

        GUI_CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
        GUI_CONTROL_FILE.write_text("", encoding="utf-8")
        self.update_live_velocity(self.vel_x_var.get(), self.vel_y_var.get(), self.vel_yaw_var.get())

        launch_args = self.build_args_with_policy(runnable_policy)
        command = [sys.executable, "-u", *launch_args]
        preview_lines = [f"{shlex.quote(sys.executable)} `"]
        for index, arg in enumerate(launch_args):
            suffix = " `" if index < len(launch_args) - 1 else ""
            preview_lines.append(f"  {arg}{suffix}")
        self.append_output(f"\n=== RUN ===\n" + "\n".join(preview_lines) + "\n\n")
        self.status_var.set("Launching local_play...")
        self.key_status_var.set("MuJoCo starting. GUI keyboard will be ready after launch.")

        def worker() -> None:
            try:
                self.proc = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    self.log_queue.put(line)
                return_code = self.proc.wait()
                self.log_queue.put(f"\n[process exited] code={return_code}\n")
            except Exception as exc:  # pragma: no cover - UI path
                self.log_queue.put(f"\n[launch failed] {exc}\n")
            finally:
                self.proc = None

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(200, self.focus_keyboard_capture)

    def stop_policy(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            self.status_var.set("No running local_play process.")
            self.key_status_var.set("No running local_play process.")
            return
        self.proc.terminate()
        self.status_var.set("Sent terminate signal to local_play.")
        self.key_status_var.set("Sent terminate signal to local_play.")

    def copy_command(self) -> None:
        preview = self.render_preview()
        self.root.clipboard_clear()
        self.root.clipboard_append(preview)
        self.status_var.set("Command copied to clipboard.")

    def flush_log_queue(self) -> None:
        wrote = False
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.append_output(line)
            wrote = True

        if wrote:
            if self.proc and self.proc.poll() is None:
                self.status_var.set("local_play is running...")
            elif not self.proc:
                self.status_var.set(f"Project root: {PROJECT_ROOT}")
                self.key_status_var.set("GUI keyboard idle. Start local_play, then click the right panel.")

        self.root.after(120, self.flush_log_queue)


def main() -> int:
    root = tk.Tk()
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    LocalPlayGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
