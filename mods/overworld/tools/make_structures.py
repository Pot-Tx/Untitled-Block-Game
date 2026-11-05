# Builds the postcard structure chunks from the source volumes below.
#
# The game reads a `BlockChunk` as a magic, a little endian version and a
# postcard document. Postcard stores the size, then the run length encoded
# `(meta, count)` pairs.

import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "assets" / "structures"


def varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def encode_chunk(size, blocks):
    runs = []
    for block in blocks:
        if runs and runs[-1][0] == block and runs[-1][1] < 0xFFFF:
            runs[-1] = (block, runs[-1][1] + 1)
        else:
            runs.append((block, 1))

    payload = bytearray()
    payload += bytes(size)
    payload += varint(len(runs))
    for block, count in runs:
        payload += varint(block)
        payload += varint(count)
    return b"STRU" + struct.pack("<I", 0) + bytes(payload)


def tree():
    size = (5, 8, 5)
    blocks = [0] * (size[0] * size[1] * size[2])

    def fill(min, max, block):
        for x in range(min[0], max[0]):
            for y in range(min[1], max[1]):
                for z in range(min[2], max[2]):
                    blocks[x * size[1] * size[2] + y * size[2] + z] = block

    fill((0, 4, 1), (5, 7, 4), 5)
    fill((1, 3, 1), (4, 8, 4), 5)
    fill((1, 4, 0), (4, 7, 5), 5)
    fill((2, 0, 2), (3, 7, 3), 4)
    return encode_chunk(size, blocks)


def camp():
    size = (3, 1, 3)
    return encode_chunk(size, [1] * 9)


if __name__ == "__main__":
    (ROOT / "tree.bin").write_bytes(tree())
    (ROOT / "camp.bin").write_bytes(camp())
    print(f"wrote {ROOT / 'tree.bin'}")
    print(f"wrote {ROOT / 'camp.bin'}")
