"""
GUI Application for Motor Imagery EEG Analysis with CSP.

Tkinter-based interface with:
 - Animated multi-channel EEG signal scrolling display
 - CSP spatial pattern visualization
 - Feature distribution and classification results
 - Exportable report figures
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from collections import deque


class EEGViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG 运动想象分析系统 — Motor Imagery Analysis with CSP")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")

        # Data state
        self.data = None          # (n_trials, n_channels, n_times)
        self.labels = None        # (n_trials,)
        self.epochs = None        # MNE Epochs object
        self.raw = None           # MNE Raw object
        self.ch_names = None
        self.times = None
        self.sfreq = None

        # CSP state
        self.csp = None
        self.features = None
        self.y_pred = None
        self.accuracy = None
        self.accuracy_std = None

        # Animation state
        self.anim = None
        self.playing = False
        self.scroll_pos = 0
        self.scroll_speed = 3  # samples per frame

        # Display settings
        self.display_channels = None
        self.n_display_ch = 10
        self.window_samples = 400

        self._build_ui()
        self._update_status("就绪 — 点击「加载数据」开始")

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # -- Top control bar --
        control_frame = tk.Frame(self.root, bg="#e8e8e8", height=50)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(control_frame, text="加载数据", command=self._load_data_thread,
                  bg="#3498db", fg="white", font=("Arial", 11, "bold"),
                  width=12, height=1).pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(control_frame, text="运行 CSP", command=self._run_csp_thread,
                  bg="#2ecc71", fg="white", font=("Arial", 11, "bold"),
                  width=12, height=1).pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(control_frame, text="导出报告图", command=self._export_figures,
                  bg="#e67e22", fg="white", font=("Arial", 11, "bold"),
                  width=12, height=1).pack(side=tk.LEFT, padx=5, pady=5)

        # Playback controls
        tk.Button(control_frame, text="▶ 播放", command=self._toggle_play,
                  bg="#9b59b6", fg="white", font=("Arial", 11),
                  width=8, height=1).pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(control_frame, text="⏸ 暂停", command=self._pause,
                  bg="#95a5a6", fg="white", font=("Arial", 11),
                  width=8, height=1).pack(side=tk.LEFT, padx=2, pady=5)

        tk.Label(control_frame, text="速度:", bg="#e8e8e8", font=("Arial", 10)).pack(side=tk.LEFT, padx=(15, 2))
        self.speed_var = tk.IntVar(value=3)
        tk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.speed_var,
                 bg="#e8e8e8", length=100, command=lambda v: setattr(self, "scroll_speed", int(v))).pack(side=tk.LEFT)

        # Status label
        self.status_label = tk.Label(control_frame, text="", bg="#e8e8e8",
                                     font=("Arial", 10), anchor=tk.W)
        self.status_label.pack(side=tk.RIGHT, padx=15, pady=5, fill=tk.X, expand=True)

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode="indeterminate", length=120)
        self.progress.pack(side=tk.RIGHT, padx=5)

        # -- Notebook (tabs) --
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: EEG Signals
        self.tab1 = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab1, text="EEG 信号浏览")
        self._build_eeg_tab()

        # Tab 2: CSP Patterns
        self.tab2 = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab2, text="CSP 空间模式")
        self._build_csp_tab()

        # Tab 3: Feature Distribution
        self.tab3 = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab3, text="特征分布")
        self._build_feature_tab()

        # Tab 4: Classification Results
        self.tab4 = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab4, text="分类结果")
        self._build_result_tab()

    # -- Tab builders --
    def _build_eeg_tab(self):
        self.fig_eeg = Figure(figsize=(13, 7), dpi=100)
        self.fig_eeg.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05, hspace=0.15)
        self.canvas_eeg = FigureCanvasTkAgg(self.fig_eeg, master=self.tab1)
        self.canvas_eeg.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar_frame = tk.Frame(self.tab1)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.canvas_eeg, toolbar_frame)

    def _build_csp_tab(self):
        self.fig_csp = Figure(figsize=(12, 8), dpi=100)
        self.canvas_csp = FigureCanvasTkAgg(self.fig_csp, master=self.tab2)
        self.canvas_csp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(self.tab2)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.canvas_csp, toolbar_frame)

    def _build_feature_tab(self):
        self.fig_features = Figure(figsize=(12, 5), dpi=100)
        self.canvas_features = FigureCanvasTkAgg(self.fig_features, master=self.tab3)
        self.canvas_features.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(self.tab3)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.canvas_features, toolbar_frame)

    def _build_result_tab(self):
        self.fig_results = Figure(figsize=(12, 5), dpi=100)
        self.canvas_results = FigureCanvasTkAgg(self.fig_results, master=self.tab4)
        self.canvas_results.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(self.tab4)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.canvas_results, toolbar_frame)

    # ------------------------------------------------------------------
    # Status & progress
    # ------------------------------------------------------------------
    def _update_status(self, msg):
        self.status_label.config(text=msg)

    def _start_progress(self):
        self.progress.start(10)

    def _stop_progress(self):
        self.progress.stop()

    # ------------------------------------------------------------------
    # Data Loading (background thread)
    # ------------------------------------------------------------------
    def _load_data_thread(self):
        self._start_progress()
        self._update_status("正在下载 Physionet EEG 运动想象数据，请稍候...")
        thread = threading.Thread(target=self._load_data, daemon=True)
        thread.start()

    def _load_data(self):
        try:
            from data_loader import load_real_data
            data, y, epochs, raw = load_real_data(subject=1)

            self.data = data
            self.labels = y
            self.epochs = epochs
            self.raw = raw
            self.ch_names = epochs.ch_names
            self.times = epochs.times * 1000  # ms
            self.sfreq = epochs.info["sfreq"]

            # Choose display channels (motor-cortex relevant)
            motor_chs = ["C3", "C1", "Cz", "C2", "C4", "FC3", "FC1", "FCz", "FC2", "FC4",
                         "CP3", "CP1", "CPz", "CP2", "CP4"]
            available = [ch for ch in motor_chs if ch in self.ch_names]
            if len(available) < self.n_display_ch:
                available = self.ch_names[:self.n_display_ch]
            self.display_channels = available[:self.n_display_ch]
            self.n_display_ch = len(self.display_channels)

            self.window_samples = int(0.5 * self.sfreq)  # 500ms window

            self.root.after(0, self._on_data_loaded)

        except Exception as e:
            self.root.after(0, lambda: self._on_error(f"数据加载失败: {e}"))

    def _on_data_loaded(self):
        self._stop_progress()
        n_trials, n_ch, n_times = self.data.shape
        n_class0 = int(np.sum(self.labels == 0))
        n_class1 = int(np.sum(self.labels == 1))
        self._update_status(
            f"已加载: {n_trials} trials, {n_ch} 导联, {n_times} 采样点, "
            f"Class 1: {n_class0}, Class 2: {n_class1}"
        )
        self._draw_eeg_signals()
        self._start_animation()
        messagebox.showinfo("加载完成", f"Physionet 运动想象数据已就绪\n"
                                       f"{n_trials} trials × {n_ch} 导联\n"
                                       f"左拳 vs 右拳")

    def _on_error(self, msg):
        self._stop_progress()
        self._update_status("错误")
        messagebox.showerror("错误", msg)

    # ------------------------------------------------------------------
    # EEG Signal Display (animated scrolling)
    # ------------------------------------------------------------------
    def _draw_eeg_signals(self):
        """Create the multi-channel EEG display axes."""
        self.fig_eeg.clear()
        n = self.n_display_ch
        self.axes_eeg = []

        for i in range(n):
            ax = self.fig_eeg.add_subplot(n, 1, i + 1)
            ax.set_facecolor("#fafafa")
            ax.tick_params(labelsize=6, colors="#888888")
            ax.set_ylabel(self.display_channels[i], fontsize=7, rotation=0,
                          labelpad=20, color="#333333")
            ax.set_yticks([])
            if i < n - 1:
                ax.set_xticks([])
            self.axes_eeg.append(ax)

        self.axes_eeg[-1].set_xlabel("Time (samples)", fontsize=8)

        # Prepare data buffer: concatenate all trials
        self.display_data = np.hstack([self.data[i] for i in range(self.data.shape[0])])
        self.display_labels = np.hstack([np.full(self.data.shape[2], self.labels[i])
                                          for i in range(len(self.labels))])

        total_len = self.display_data.shape[1]
        self.axes_eeg[-1].set_xlim(0, self.window_samples)
        self.axes_eeg[-1].set_xlabel("Time (samples)", fontsize=8)

        # Initialize lines for each channel
        self.eeg_lines = []
        colors_tab10 = plt.cm.tab10.colors

        for i, ax in enumerate(self.axes_eeg):
            ch_idx = self.ch_names.index(self.display_channels[i])
            data_slice = self.display_data[ch_idx, :self.window_samples]
            y_min = float(np.min(self.display_data[ch_idx, :]))
            y_max = float(np.max(self.display_data[ch_idx, :]))
            margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
            ax.set_ylim(y_min - margin, y_max + margin)

            colors_for_samples = self._sample_colors(0, self.window_samples)
            line = ax.plot(range(self.window_samples), data_slice,
                           color=colors_tab10[i % 10], linewidth=0.7)[0]
            self.eeg_lines.append(line)

        self.fig_eeg.suptitle("多导联 EEG 信号 — 动态浏览", fontsize=12, fontweight="bold",
                              color="#2c3e50")
        self.canvas_eeg.draw()

    def _sample_colors(self, start, end):
        """Map class labels to colors for a segment."""
        colors = []
        for j in range(start, min(end, len(self.display_labels))):
            if self.display_labels[j] == 0:
                colors.append("#3498db")
            else:
                colors.append("#e74c3c")
        return colors

    def _animate_eeg(self, frame):
        """Animation callback: advance the scrolling window."""
        if not self.playing:
            return self.eeg_lines

        self.scroll_pos += self.scroll_speed
        total_len = self.display_data.shape[1]

        if self.scroll_pos + self.window_samples >= total_len:
            self.scroll_pos = 0  # loop

        start = self.scroll_pos
        end = start + self.window_samples

        for i, line in enumerate(self.eeg_lines):
            ch_idx = self.ch_names.index(self.display_channels[i])
            line.set_ydata(self.display_data[ch_idx, start:end])

        # Update x-axis
        for ax in self.axes_eeg:
            ax.set_xlim(start, end)

        # Update status
        self._update_status(
            f"显示中: sample {start}-{end} / {total_len} | "
            f"速度: {self.scroll_speed}x | "
            f"Trials: {self.data.shape[0]}, 导联: {self.data.shape[1]}"
        )

        return self.eeg_lines

    def _start_animation(self):
        if self.anim is not None:
            self.anim.event_source.stop()
        self.playing = True
        self.anim = FuncAnimation(self.fig_eeg, self._animate_eeg, interval=40,
                                   blit=False, cache_frame_data=False)
        self.canvas_eeg.draw()

    def _toggle_play(self):
        if self.data is None:
            messagebox.showwarning("提示", "请先加载数据")
            return
        self.playing = not self.playing

    def _pause(self):
        self.playing = False

    # ------------------------------------------------------------------
    # CSP Analysis (background thread)
    # ------------------------------------------------------------------
    def _run_csp_thread(self):
        if self.data is None:
            messagebox.showwarning("提示", "请先加载数据")
            return

        self._start_progress()
        self._update_status("正在运行 CSP 分析...")
        thread = threading.Thread(target=self._run_csp, daemon=True)
        thread.start()

    def _run_csp(self):
        try:
            from csp import CSP
            from analysis import classify_csp_features

            csp = CSP(n_components=4)
            features = csp.fit_transform(self.data, self.labels)

            accuracy, accuracy_std, clf, y_pred = classify_csp_features(features, self.labels)

            self.csp = csp
            self.features = features
            self.y_pred = y_pred
            self.accuracy = accuracy
            self.accuracy_std = accuracy_std

            self.root.after(0, self._on_csp_done)

        except Exception as e:
            self.root.after(0, lambda: self._on_error(f"CSP 分析失败: {e}"))

    def _on_csp_done(self):
        self._stop_progress()
        self._update_status(
            f"CSP 完成! 分类准确率: {self.accuracy * 100:.1f}% ± {self.accuracy_std * 100:.1f}%"
        )
        self._draw_csp_patterns()
        self._draw_feature_distribution()
        self._draw_classification_results()
        self.notebook.select(self.tab2)
        messagebox.showinfo("CSP 完成",
                           f"5-Fold CV 准确率: {self.accuracy * 100:.1f}% ± {self.accuracy_std * 100:.1f}%\n"
                           f"请查看各标签页的分析结果")

    # ------------------------------------------------------------------
    # CSP Patterns Plot
    # ------------------------------------------------------------------
    def _draw_csp_patterns(self):
        from analysis import plot_csp_patterns, plot_eigenvalue_spectrum

        self.fig_csp.clear()

        # Left: spatial patterns
        gs = self.fig_csp.add_gridspec(1, 2, width_ratios=[1, 1])
        ax_patterns = self.fig_csp.add_subplot(gs[0, 0])

        patterns = self.csp.get_spatial_patterns()
        n_comp = patterns.shape[1]
        n_ch = len(self.ch_names)
        m = n_comp // 2

        colors_map = plt.cm.RdBu_r
        component_rows = 2 if m > 1 else 1

        for i in range(n_comp):
            vals = patterns[:, i]
            vmax = np.max(np.abs(vals)) if np.max(np.abs(vals)) > 0 else 1.0
            colors = [colors_map(0.5 + 0.5 * v / vmax) for v in vals]

            ax_patterns.barh(range(n_ch), vals, color=colors, edgecolor="gray", linewidth=0.3,
                             label=f"C{i + 1}")
        ax_patterns.set_yticks(range(n_ch))
        ax_patterns.set_yticklabels(self.ch_names, fontsize=6)
        ax_patterns.axvline(x=0, color="black", linewidth=0.8)
        ax_patterns.set_title("CSP 空间模式 (所有成分)", fontsize=11)
        ax_patterns.set_xlabel("权重", fontsize=8)
        ax_patterns.legend(fontsize=7, loc="lower right")

        # Right: eigenvalue spectrum
        ax_eig = self.fig_csp.add_subplot(gs[0, 1])
        eigvals = self.csp.eigvals_
        n_ev = len(eigvals)
        ax_eig.plot(range(1, n_ev + 1), eigvals, "o-", markersize=3, color="#2c3e50", linewidth=1)
        ax_eig.axvline(x=m + 0.5, color="red", linestyle="--", alpha=0.5)
        ax_eig.axvline(x=n_ev - m + 0.5, color="red", linestyle="--", alpha=0.5,
                       label="Selected components")
        ax_eig.set_xlabel("Component index", fontsize=8)
        ax_eig.set_ylabel("Eigenvalue", fontsize=8)
        ax_eig.set_title("CSP 特征值谱", fontsize=11)
        ax_eig.legend(fontsize=7)
        ax_eig.grid(True, alpha=0.3)

        self.fig_csp.suptitle("CSP 空间模式分析", fontsize=12, fontweight="bold", color="#2c3e50")
        self.fig_csp.tight_layout()
        self.canvas_csp.draw()

    # ------------------------------------------------------------------
    # Feature Distribution Plot
    # ------------------------------------------------------------------
    def _draw_feature_distribution(self):
        from analysis import plot_feature_distribution
        self.fig_features.clear()
        plot_feature_distribution(self.features, self.labels)
        self.fig_features.suptitle("CSP 特征分布", fontsize=12, fontweight="bold", color="#2c3e50")
        self.fig_features.tight_layout()
        self.canvas_features.draw()

    # ------------------------------------------------------------------
    # Classification Results Plot
    # ------------------------------------------------------------------
    def _draw_classification_results(self):
        from analysis import plot_classification_results
        self.fig_results.clear()
        plot_classification_results(self.accuracy, self.accuracy_std,
                                     self.labels, self.y_pred)
        self.fig_results.suptitle("分类结果 (CSP + LDA)", fontsize=12, fontweight="bold",
                                  color="#2c3e50")
        self.fig_results.tight_layout()
        self.canvas_results.draw()

    # ------------------------------------------------------------------
    # Export Report Figures
    # ------------------------------------------------------------------
    def _export_figures(self):
        if self.csp is None or self.features is None:
            messagebox.showwarning("提示", "请先运行 CSP 分析")
            return

        output_dir = filedialog.askdirectory(title="选择报告图输出目录")
        if not output_dir:
            return

        try:
            from analysis import generate_all_report_figures
            generate_all_report_figures(
                self.csp, self.features, self.labels, self.y_pred,
                self.accuracy, self.accuracy_std, self.epochs, output_dir
            )
            self._update_status(f"报告图已导出到: {output_dir}")
            messagebox.showinfo("导出完成",
                               f"5 张报告图已保存到:\n{output_dir}\n\n"
                               "01_raw_eeg_overview.png\n"
                               "02_csp_spatial_patterns.png\n"
                               "03_csp_eigenvalue_spectrum.png\n"
                               "04_feature_distribution.png\n"
                               "05_classification_results.png")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))
