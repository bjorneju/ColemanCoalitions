"""
coleman_coalitions/io.py

File and screen output utilities for analysis results.

Functions here accept the plain-dict format produced by run_full_analysis()
and write or print human-readable summaries using the DESCRIPTIONS lookup from core.py.
"""
from __future__ import annotations

import os
import time

import numpy as np
from numpy import ndarray

from .core import DESCRIPTIONS


def _format_value(data: object) -> str:
    """Format a scalar or ndarray for display (4 decimal places)."""
    arr = np.asarray(data)
    if arr.size == 1:
        return f'{float(arr):.4f}'
    lines: list[str] = []
    if arr.ndim == 1:
        lines.append('  ' + '  '.join(f'{v:8.4f}' for v in arr))
    else:
        for row in arr:
            lines.append('  ' + '  '.join(f'{v:8.4f}' for v in np.asarray(row).flatten()))
    return '\n'.join(lines)


def _write_section(
    f: object,
    key: str,
    value: object,
    descriptions: dict[str, str] | None = None,
) -> None:
    """Write one variable (heading + formatted value) to file handle f."""
    desc = (descriptions or DESCRIPTIONS).get(key, key)
    formatted = _format_value(value)
    if '\n' in formatted:
        # Multi-line: print label on its own line, then the matrix
        print(f'{desc}:', file=f)
        print(formatted, file=f)
    else:
        # Scalar or 1-D: print on one line
        print(f'{desc}: {formatted}', file=f)
    print('', file=f)  # blank line separator


def write_analysis(
    inputs: dict,
    filepath: str,
    title: str,
    order: list[str] | None = None,
    descriptions: dict[str, str] | None = None,
) -> str:
    """Write analysis results to a timestamped text file.

    Parameters
    ----------
    inputs : dict
        Analysis outputs (plain ndarray values).
    filepath : str
        Output directory (created if it does not exist).
    title : str
        Filename prefix; the file is named ``YYYYMMDD_<title>.txt``.
    order : list[str] or None
        Keys to write, in order.  Defaults to all keys in ``inputs``.
    descriptions : dict or None
        Override variable descriptions (falls back to DESCRIPTIONS).

    Returns
    -------
    str  -- path to the written file
    """
    os.makedirs(filepath, exist_ok=True)
    filename = os.path.join(filepath, time.strftime('%Y%m%d') + '_' + title + '.txt')
    keys = order if order is not None else list(inputs.keys())
    header = (
        f"{'*' * 48}\n  {title}\n"
        f"  Created: {time.strftime('%d/%m/%Y %H:%M:%S')}\n{'*' * 48}\n"
    )
    with open(filename, 'w') as f:
        print(header, file=f)
        for key in keys:
            if key in inputs:
                _write_section(f, key, inputs[key], descriptions)
        print(header, file=f)
    return filename


def print_analysis(
    inputs: dict,
    order: list[str] | None = None,
    descriptions: dict[str, str] | None = None,
) -> None:
    """Print analysis results to stdout.

    Parameters
    ----------
    inputs : dict
    order : list[str] or None  -- keys to print (defaults to all keys)
    descriptions : dict or None
    """
    keys = order if order is not None else list(inputs.keys())
    for key in keys:
        if key in inputs:
            desc = (descriptions or DESCRIPTIONS).get(key, key)
            formatted = _format_value(inputs[key])
            if '\n' in formatted:
                print(f'{desc}:')
                print(formatted)
            else:
                print(f'{desc}: {formatted}')
            print()
