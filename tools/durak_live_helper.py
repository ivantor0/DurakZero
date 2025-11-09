#!/usr/bin/env python3
"""Real-time DurakZero assistant for live multiplayer games."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, Optional

import torch

from douzero.evaluation.simulation import load_model
from douzero.live import LiveDurakTracker


def _tail_file(path: Path) -> Iterable[str]:
    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line.rstrip('\n')


def _capture_pyshark(interface: str, display_filter: Optional[str]) -> Iterable[str]:
    try:
        import pyshark
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pyshark is not installed. Install it with `pip install pyshark` "
            "or use --source file to read from a log."
        ) from exc

    capture = pyshark.LiveCapture(interface=interface, display_filter=display_filter)
    for packet in capture.sniff_continuously():
        if 'DATA' not in packet:
            continue
        try:
            hex_payload = str(packet.DATA.data)
            text = bytes.fromhex(hex_payload).decode('utf-8', errors='ignore')
        except Exception:
            continue
        for line in text.splitlines():
            yield line


def _resolve_device(device_arg: str) -> torch.device:
    lower = device_arg.lower()
    if lower.startswith('cpu'):
        return torch.device('cpu')
    if lower.startswith('cuda'):
        if ':' in lower:
            index = lower.split(':', 1)[1] or '0'
        else:
            index = '0'
        return torch.device(f'cuda:{index}')
    if lower.isdigit():
        return torch.device(f'cuda:{lower}')
    return torch.device(device_arg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (model.tar).')
    parser.add_argument('--device', default='cpu', help='Device for inference: cpu, cuda, or GPU index.')
    parser.add_argument('--player-id', type=int, default=None, help='Override detected player seat (0 or 1).')
    parser.add_argument('--source', choices=['pyshark', 'file'], default='pyshark',
                        help='How to ingest live packets.')
    parser.add_argument('--interface', default='waydroid0', help='Network interface for pyshark capture.')
    parser.add_argument('--ip-filter', default=None, help='Wireshark display filter, e.g. "ip.addr == 65.21.92.166".')
    parser.add_argument('--log-file', type=Path, help='When --source=file, tail this UTF-8 log.')
    args = parser.parse_args()

    model = load_model(args.checkpoint, device=args.device)

    tracker_device = _resolve_device(args.device)
    tracker = LiveDurakTracker(model=model, device=tracker_device,
                               player_id=args.player_id, verbose=True)

    if args.source == 'file':
        if args.log_file is None:
            parser.error('--log-file is required when --source=file')
        source_iter = _tail_file(args.log_file)
        print(f"Tail {args.log_file} for new packets. Press Ctrl+C to stop.")
    else:
        source_iter = _capture_pyshark(args.interface, args.ip_filter)
        print(f"Listening on {args.interface} (filter: {args.ip_filter or 'none'}). Press Ctrl+C to stop.")

    try:
        for line in source_iter:
            tracker.process_raw_line(line)
    except KeyboardInterrupt:
        print('\nStopping live helper.')


if __name__ == '__main__':
    main()
