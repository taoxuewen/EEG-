"""
Analysis and report figure generation for motor imagery EEG + CSP.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedKFold


def classify_csp_features(features, y):
    """
    Train LDA classifier on CSP features with 5-fold CV.
    Returns accuracy, trained model, prediction on training data.
    """
    clf = LDA()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, features, y, cv=cv, scoring="accuracy")

    clf.fit(features, y)
    y_pred = clf.predict(features)

    return np.mean(scores), np.std(scores), clf, y_pred


def plot_raw_eeg_overview(epochs, save_path=None):
    """Multi-channel EEG time series overview. Report-ready."""
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    times = epochs.times * 1000  # ms
    ch_names = epochs.ch_names
    n_ch = min(16, len(ch_names))
    n_show = min(20, data.shape[0])

    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Motor Imagery EEG — Raw Traces (Sample Trials)", fontsize=14, fontweight="bold")

    colors = ["#1f77b4", "#ff7f0e"]
    for ch_idx in range(n_ch):
        ax = axes[ch_idx]
        for trial_idx in range(n_show):
            ax.plot(times, data[trial_idx, ch_idx, :], alpha=0.15,
                    color=colors[trial_idx % 2], linewidth=0.5)
        ax.set_ylabel(ch_names[ch_idx], fontsize=7, rotation=0, labelpad=25)
        ax.tick_params(labelsize=6)
        ax.axvline(x=1000, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.set_yticks([])

    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    axes[0].legend(["Class 1 (Left)", "Class 2 (Right)"], fontsize=8, loc="upper right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_csp_patterns(csp, ch_names, save_path=None):
    """CSP spatial patterns as topoplot-like bar charts. Report-ready."""
    patterns = csp.get_spatial_patterns()  # (n_channels, n_components)
    n_comp = patterns.shape[1]
    n_ch = len(ch_names)
    m = n_comp // 2

    fig, axes = plt.subplots(2, m, figsize=(4 * m, 8))
    fig.suptitle("CSP Spatial Patterns", fontsize=14, fontweight="bold")

    colors_map = plt.cm.RdBu_r
    for i in range(n_comp):
        row = 0 if i < m else 1
        col = i if i < m else i - m

        if isinstance(axes, np.ndarray):
            ax = axes[row, col] if m > 1 else axes[row]
        else:
            ax = axes

        vals = patterns[:, i]
        vmax = np.max(np.abs(vals))
        colors = [colors_map(0.5 + 0.5 * v / vmax) if vmax > 0 else colors_map(0.5) for v in vals]

        ax.barh(range(n_ch), vals, color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_yticks(range(n_ch))
        ax.set_yticklabels(ch_names, fontsize=7)
        ax.axvline(x=0, color="black", linewidth=0.8)
        class_label = "Class 1 (Left)" if i < m else "Class 2 (Right)"
        ax.set_title(f"CSP Component {i + 1}\n({class_label})", fontsize=10)
        ax.set_xlabel("Weight", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_eigenvalue_spectrum(csp, save_path=None):
    """CSP eigenvalue spectrum showing discriminability. Report-ready."""
    eigvals = csp.eigvals_
    m = csp.n_components // 2
    n = len(eigvals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CSP Eigenvalue Analysis", fontsize=14, fontweight="bold")

    # Full spectrum
    ax = axes[0]
    ax.plot(range(1, n + 1), eigvals, "o-", markersize=4, color="#2c3e50", linewidth=1)
    ax.axvline(x=m + 0.5, color="red", linestyle="--", alpha=0.6, label="Selected components")
    ax.axvline(x=n - m + 0.5, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Sorted Eigenvalue Spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Zoomed selected components
    ax = axes[1]
    selected_evals = np.concatenate([eigvals[:m], eigvals[-m:]])
    component_labels = [f"F{i + 1}" for i in range(m)] + [f"F{n - m + i + 1}" for i in range(m)]
    bars = ax.bar(range(len(selected_evals)), selected_evals, color=["#3498db"] * m + ["#e74c3c"] * m,
                  edgecolor="gray", linewidth=0.5)
    ax.set_xticks(range(len(selected_evals)))
    ax.set_xticklabels(component_labels, fontsize=8)
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Selected CSP Component Eigenvalues")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_feature_distribution(features, y, save_path=None):
    """CSP feature space scatter plot. Report-ready."""
    n_comp = features.shape[1]
    m = n_comp // 2

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig)
    fig.suptitle("CSP Feature Distribution", fontsize=14, fontweight="bold")

    # Scatter: first two components (one from each class discriminator)
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ["#3498db" if lab == 0 else "#e74c3c" for lab in y]
    ax1.scatter(features[:, 0], features[:, m], c=colors, alpha=0.7, edgecolors="black", linewidth=0.5, s=60)
    ax1.set_xlabel(f"Feature 1 (Class 1 discriminator)", fontsize=9)
    ax1.set_ylabel(f"Feature {m + 1} (Class 2 discriminator)", fontsize=9)
    ax1.set_title("CSP Feature Space", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(["Class 1 (Left)", "Class 2 (Right)"], fontsize=8)

    # Box plot of all features
    ax2 = fig.add_subplot(gs[0, 1])
    feature_data = []
    feature_labels = []
    for i in range(n_comp):
        for lab_val, lab_name in [(0, "Left"), (1, "Right")]:
            idx = np.where(y == lab_val)[0]
            feature_data.append(features[idx, i])
            feature_labels.append(f"{i + 1}\n{lab_name}")

    bp = ax2.boxplot(feature_data, patch_artist=True, widths=0.6)
    for j, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#3498db" if "Left" in feature_labels[j] else "#e74c3c")
        patch.set_alpha(0.6)
    ax2.set_xticklabels(feature_labels, fontsize=6, rotation=45)
    ax2.set_ylabel("Log-Variance", fontsize=9)
    ax2.set_title("Feature Distributions by Class", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_classification_results(accuracy, accuracy_std, y_true, y_pred, save_path=None):
    """Classification accuracy bar chart + confusion matrix. Report-ready."""
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig)
    fig.suptitle("Motor Imagery Classification Results (CSP + LDA)", fontsize=14, fontweight="bold")

    # Accuracy bar
    ax1 = fig.add_subplot(gs[0, 0])
    bar = ax1.bar(["CSP + LDA"], [accuracy * 100], yerr=[accuracy_std * 100],
                  color="#2ecc71", edgecolor="black", linewidth=1, capsize=10, width=0.4)
    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Chance level (50%)")
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.set_title(f"5-Fold CV Accuracy: {accuracy * 100:.1f} ± {accuracy_std * 100:.1f}%", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Left Fist", "Right Fist"])
    disp.plot(ax=ax2, cmap="Blues", colorbar=False)
    ax2.set_title("Confusion Matrix (Training Set)", fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def generate_all_report_figures(csp, features, y, y_pred, accuracy, accuracy_std, epochs, output_dir):
    """Generate all report-quality figures and save to output_dir."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig1 = plot_raw_eeg_overview(epochs, save_path=os.path.join(output_dir, "01_raw_eeg_overview.png"))
    fig2 = plot_csp_patterns(csp, epochs.ch_names, save_path=os.path.join(output_dir, "02_csp_spatial_patterns.png"))
    fig3 = plot_eigenvalue_spectrum(csp, save_path=os.path.join(output_dir, "03_csp_eigenvalue_spectrum.png"))
    fig4 = plot_feature_distribution(features, y, save_path=os.path.join(output_dir, "04_feature_distribution.png"))
    fig5 = plot_classification_results(accuracy, accuracy_std, y, y_pred,
                                        save_path=os.path.join(output_dir, "05_classification_results.png"))

    for fig in [fig1, fig2, fig3, fig4, fig5]:
        plt.close(fig)

    print(f"Report figures saved to: {output_dir}")
    return output_dir
