"""
Download and preprocess Physionet EEG Motor Imagery dataset via MNE.
"""

import numpy as np
from scipy import signal as scipy_signal


def download_eegbci(subjects=None):
    """
    Download Physionet EEG Motor Movement/Imagery dataset.

    Uses records 3,7,11 (left vs right fist motor imagery).
    Returns list of Raw objects (one per subject, concatenated runs).
    """
    import mne
    from mne.datasets import eegbci

    if subjects is None:
        subjects = [1]

    raw_list = []
    for subj in subjects:
        fnames = eegbci.load_data(subj, [3, 7, 11], update_path=True)
        raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames]
        # Standardize each raw individually so channel names match
        for r in raws:
            eegbci.standardize(r)
        # Concatenate multiple runs for more trials
        raw = mne.concatenate_raws(raws)
        raw_list.append(raw)

    return raw_list


def preprocess_pipeline(raw, l_freq=8.0, h_freq=30.0):
    """
    Preprocess raw EEG for motor imagery CSP analysis.

    Steps:
    1. Pick EEG channels only
    2. Bandpass filter (mu + beta rhythms, default 8-30 Hz)
    3. Extract motor imagery epochs

    Returns epochs array (n_epochs, n_channels, n_times), labels, and the Epochs object.
    """
    import mne

    # Pick EEG channels
    raw.pick_types(eeg=True, exclude="bads")

    # Bandpass filter
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose=False)

    # Find events
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Motor imagery events: T1=left fist, T2=right fist
    mi_events = {}
    if "T1" in event_id:
        mi_events["left_fist"] = event_id["T1"]
    if "T2" in event_id:
        mi_events["right_fist"] = event_id["T2"]

    if len(mi_events) < 2:
        # Fallback: find any two event types
        keys = list(event_id.keys())[:2]
        mi_events = {k: event_id[k] for k in keys}

    # Extract epochs: 0.5s before to 2.5s after cue
    epochs = mne.Epochs(
        raw,
        events,
        event_id=mi_events,
        tmin=0.5,
        tmax=2.5,
        baseline=(0.5, 1.0),
        preload=True,
        verbose=False,
    )

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1]

    # Binarize labels: 0 for first class, 1 for second
    unique_labels = np.unique(labels)
    y = np.where(labels == unique_labels[0], 0, 1)

    return data, y, epochs


def load_real_data(subject=1):
    """
    Load and preprocess real motor imagery EEG data.
    Returns data, labels, epochs, raw.
    """
    raw_list = download_eegbci([subject])
    raw = raw_list[0]
    data, y, epochs = preprocess_pipeline(raw)
    return data, y, epochs, raw
