#!/usr/bin/env python3
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python3 npz_info.py <file.npz>")
    sys.exit(1)

filename = sys.argv[1]

data = np.load(filename, allow_pickle=True)

print(f"\nFile: {filename}")
print("-" * 40)

for key in data.files:
    arr = data[key]
    print(f"{key:25s} shape={arr.shape}   dtype={arr.dtype}")

print("-" * 40)
print("Done.\n")
