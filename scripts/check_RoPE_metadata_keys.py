#!/usr/bin/env python3

import mmap
import struct
import sys


def list_matching_metadata_keys(path: str, substrings: tuple[str, ...]) -> list[str]:
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        size = len(mm)
        pos = 0

        def need(n: int) -> None:
            nonlocal pos
            if pos + n > size:
                raise EOFError(f"need {n} bytes at {pos}/{size}")

        def u32() -> int:
            nonlocal pos
            need(4)
            v = struct.unpack_from("<I", mm, pos)[0]
            pos += 4
            return v

        def u64() -> int:
            nonlocal pos
            need(8)
            v = struct.unpack_from("<Q", mm, pos)[0]
            pos += 8
            return v

        def skip(n: int) -> None:
            nonlocal pos
            need(n)
            pos += n

        def read_str() -> str:
            nonlocal pos
            n = u64()
            need(n)
            s = mm[pos : pos + n].decode("utf-8", "replace")
            pos += n
            return s

        need(4)
        magic = mm[pos : pos + 4]
        pos += 4
        if magic != b"GGUF":
            raise ValueError("bad magic (not GGUF)")

        _version = u32()
        _n_tensors = u64()
        n_kv = u64()

        # GGUF value types
        U8, I8, U16, I16, U32T, I32T, F32, BOOL, STR, ARR, U64T, I64T, F64 = range(13)
        sz = {
            U8: 1,
            I8: 1,
            BOOL: 1,
            U16: 2,
            I16: 2,
            U32T: 4,
            I32T: 4,
            F32: 4,
            U64T: 8,
            I64T: 8,
            F64: 8,
        }

        def skip_value(t: int) -> None:
            if t in sz:
                skip(sz[t])
            elif t == STR:
                skip(u64())
            elif t == ARR:
                et = u32()
                n = u64()
                if et == STR:
                    for _ in range(n):
                        skip(u64())
                else:
                    if et not in sz:
                        raise ValueError(f"unknown array elem type {et}")
                    skip(sz[et] * n)
            else:
                raise ValueError(f"unknown value type {t}")

        out: list[str] = []
        for _ in range(n_kv):
            k = read_str()
            t = u32()
            if any(s in k for s in substrings):
                out.append(k)
            skip_value(t)

        return out


def main() -> int:
    if len(sys.argv) == 1:
        print(f"usage: {sys.argv[0]} <model.gguf> [substr ...]", file=sys.stderr)
        return 2

    path = sys.argv[1]
    substrings = tuple(sys.argv[2:]) or ("rope", "bias")

    keys = list_matching_metadata_keys(path, substrings)
    for k in keys:
        print(k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

