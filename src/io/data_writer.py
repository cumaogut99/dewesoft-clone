"""
Data I/O — save and load recordings.
Supports HDF5 (default, efficient) and CSV (portable).
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import h5py
import csv
from datetime import datetime

from ..core.channel import Channel


def save_hdf5(channels: List[Channel], path: str | Path) -> None:
    """Save all channel data to an HDF5 file."""
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["created"] = datetime.now().isoformat()
        f.attrs["version"] = "1.0"
        for ch in channels:
            ts, data = ch.get_data()
            if len(data) == 0:
                continue
            grp = f.create_group(f"channel_{ch.id:03d}")
            grp.attrs["name"] = ch.name
            grp.attrs["unit"] = ch.unit
            grp.attrs["sample_rate"] = ch.sample_rate
            grp.attrs["color"] = ch.color
            grp.create_dataset("timestamps", data=ts, compression="gzip")
            grp.create_dataset("data", data=data, compression="gzip")


def load_hdf5(path: str | Path) -> list[dict]:
    """Load channel recordings from HDF5. Returns list of dicts."""
    path = Path(path)
    results = []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            grp = f[key]
            results.append({
                "name": grp.attrs.get("name", key),
                "unit": grp.attrs.get("unit", ""),
                "sample_rate": float(grp.attrs.get("sample_rate", 1000.0)),
                "color": grp.attrs.get("color", "#ffffff"),
                "timestamps": np.array(grp["timestamps"]),
                "data": np.array(grp["data"]),
            })
    return results


def save_csv(channels: List[Channel], path: str | Path) -> None:
    """Save channel data to CSV. One column per channel plus a time column."""
    path = Path(path)
    data_by_ch = []
    for ch in channels:
        ts, data = ch.get_data()
        data_by_ch.append((ch, ts, data))

    if not data_by_ch:
        return

    # Align to the channel with the most samples
    max_len = max(len(d) for _, _, d in data_by_ch)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Time (s)"] + [f"{ch.name} ({ch.unit})" for ch, _, _ in data_by_ch]
        writer.writerow(header)
        for i in range(max_len):
            row_ts = data_by_ch[0][1][i] if i < len(data_by_ch[0][1]) else ""
            row = [row_ts] + [
                float(data[i]) if i < len(data) else ""
                for _, _, data in data_by_ch
            ]
            writer.writerow(row)


def load_csv(path: str | Path) -> list[dict]:
    """Load CSV file saved by save_csv. Returns list of channel dicts."""
    path = Path(path)
    results: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    if len(headers) < 2:
        return results

    timestamps = np.array([float(r[0]) for r in rows if r[0] != ""])

    for col_idx, header in enumerate(headers[1:], start=1):
        name = header.split("(")[0].strip()
        unit = header.split("(")[1].rstrip(")") if "(" in header else ""
        data = np.array([float(r[col_idx]) for r in rows if col_idx < len(r) and r[col_idx] != ""])
        results.append({"name": name, "unit": unit, "timestamps": timestamps[:len(data)], "data": data})

    return results
