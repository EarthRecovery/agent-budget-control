#!/usr/bin/env python3
"""
Dense-only retrieval server for Search training.
Provides E5 embeddings + FAISS dense indexing.

This code is adapted from the RLLM project:
  https://github.com/rllm-org/rllm
  License: Apache-2.0

Usage:
    python server.py --data_dir /projects/bflz/searchr1_data/search_data/prebuilt_indices --port 8000
"""

import argparse
import json
import mmap
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

import torch
import faiss
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer


DEFAULT_SEARCHR1_DATA_ROOT = os.environ.get(
    "SEARCHR1_DATA_ROOT",
    "/projects/bflz/searchr1_data",
)
DEFAULT_INDEX_DIR = os.path.join(DEFAULT_SEARCHR1_DATA_ROOT, "search_data", "prebuilt_indices")


class LazyJsonStringArray:
    """Memory-efficient random access over a JSON array of strings."""

    OFFSETS_SUFFIX = ".offsets.u64"
    _BUFFER_SIZE = 1_000_000
    _WHITESPACE = {9, 10, 13, 32}

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offsets_path = self.path.with_name(self.path.name + self.OFFSETS_SUFFIX)
        self._ensure_offsets()

        self._file = self.path.open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._offsets = np.memmap(self.offsets_path, dtype=np.uint64, mode="r")
        if self._offsets.size == 0:
            raise ValueError(f"Offset index is empty: {self.offsets_path}")

    def __len__(self) -> int:
        return max(0, int(self._offsets.size) - 1)

    def __getitem__(self, idx: int) -> str:
        idx = int(idx)
        size = len(self)
        if idx < 0:
            idx += size
        if idx < 0 or idx >= size:
            raise IndexError(idx)

        start = int(self._offsets[idx])
        end = int(self._offsets[idx + 1])
        while end > start and self._mmap[end - 1] in self._WHITESPACE.union({44}):
            end -= 1
        return json.loads(self._mmap[start:end])

    def close(self) -> None:
        self._offsets._mmap.close()
        self._mmap.close()
        self._file.close()

    def _ensure_offsets(self) -> None:
        if self._offsets_are_fresh():
            return
        self._build_offsets()

    def _offsets_are_fresh(self) -> bool:
        if not self.offsets_path.exists():
            return False
        if self.offsets_path.stat().st_size % np.dtype(np.uint64).itemsize != 0:
            return False
        return self.offsets_path.stat().st_mtime >= self.path.stat().st_mtime

    def _build_offsets(self) -> None:
        tmp_path = self.offsets_path.with_suffix(self.offsets_path.suffix + ".tmp")
        print(f"Building lazy corpus offsets at {self.offsets_path} ...")

        with self.path.open("rb") as f, tmp_path.open("wb") as out:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                pos = self._skip_whitespace(mm, 0)
                if pos >= len(mm) or mm[pos] != ord("["):
                    raise ValueError(f"{self.path} is not a JSON array")
                pos += 1

                buffer = []
                item_count = 0
                while True:
                    pos = self._skip_whitespace(mm, pos)
                    if pos >= len(mm):
                        raise ValueError(f"Unexpected EOF while parsing {self.path}")
                    if mm[pos] == ord("]"):
                        buffer.append(pos)
                        self._flush_offsets(buffer, out)
                        break

                    buffer.append(pos)
                    item_count += 1
                    pos = self._scan_json_string(mm, pos)
                    pos = self._skip_whitespace(mm, pos)
                    if pos >= len(mm):
                        raise ValueError(f"Unexpected EOF after item {item_count} in {self.path}")

                    if mm[pos] == ord(","):
                        pos += 1
                    elif mm[pos] == ord("]"):
                        buffer.append(pos)
                        self._flush_offsets(buffer, out)
                        break
                    else:
                        raise ValueError(f"Unexpected byte {mm[pos]!r} at position {pos} in {self.path}")

                    if len(buffer) >= self._BUFFER_SIZE:
                        self._flush_offsets(buffer, out)
            finally:
                mm.close()

        tmp_path.replace(self.offsets_path)
        offsets_count = self.offsets_path.stat().st_size // np.dtype(np.uint64).itemsize
        print(f"Built lazy corpus offsets for {max(0, offsets_count - 1)} documents")

    @staticmethod
    def _flush_offsets(buffer: list[int], fh) -> None:
        if not buffer:
            return
        np.asarray(buffer, dtype=np.uint64).tofile(fh)
        buffer.clear()

    @classmethod
    def _skip_whitespace(cls, mm: mmap.mmap, pos: int) -> int:
        size = len(mm)
        while pos < size and mm[pos] in cls._WHITESPACE:
            pos += 1
        return pos

    @staticmethod
    def _scan_json_string(mm: mmap.mmap, pos: int) -> int:
        if mm[pos] != ord('"'):
            raise ValueError(f"Expected JSON string at position {pos}")

        pos += 1
        while True:
            quote_pos = mm.find(b'"', pos)
            if quote_pos == -1:
                raise ValueError("Unterminated JSON string")

            backslash_count = 0
            scan_pos = quote_pos - 1
            while scan_pos >= 0 and mm[scan_pos] == ord("\\"):
                backslash_count += 1
                scan_pos -= 1

            if backslash_count % 2 == 0:
                return quote_pos + 1
            pos = quote_pos + 1


class LocalRetriever:
    """Dense-only retrieval system using FAISS."""

    def __init__(self, data_dir: str, device: str = "cpu", gpu_memory_limit_mb: int = 6144):
        self.data_dir = Path(data_dir)
        self.corpus = []
        self.dense_index = None
        self._lock = threading.Lock()

        if device.startswith("cuda"):
            dev_idx = int(device.split(":")[-1]) if ":" in device else 0
            fraction = gpu_memory_limit_mb / (torch.cuda.get_device_properties(dev_idx).total_memory / 1024**2)
            fraction = min(fraction, 1.0)
            torch.cuda.set_per_process_memory_fraction(fraction, dev_idx)
            print(f"GPU memory limit: {gpu_memory_limit_mb}MB (fraction={fraction:.4f}) on cuda:{dev_idx}")

        self.encoder = SentenceTransformer("intfloat/e5-base-v2", device=device)
        print(f"E5 encoder loaded on device: {device}")

        self._load_data()

    def _load_data(self):
        """Load corpus and dense index from data directory."""
        print(f"Loading data from {self.data_dir}")

        # Load corpus
        corpus_file = self.data_dir / "corpus.json"
        self.corpus = LazyJsonStringArray(corpus_file)
        print(f"Loaded corpus with {len(self.corpus)} documents")

        # Load dense index
        dense_index_file = self.data_dir / "e5_Flat.index"
        mmap_flags = getattr(faiss, "IO_FLAG_MMAP", 0) | getattr(faiss, "IO_FLAG_READ_ONLY", 0)
        try:
            self.dense_index = faiss.read_index(str(dense_index_file), mmap_flags)
            print("Loaded dense index with FAISS mmap/read-only flags")
        except Exception as exc:
            print(f"FAISS mmap load unavailable ({exc}); falling back to standard read_index")
            self.dense_index = faiss.read_index(str(dense_index_file))
        print(f"Loaded dense index with {self.dense_index.ntotal} vectors")

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Dense retrieval using FAISS. Lock serializes GPU encoding to prevent memory bloat."""
        with self._lock:
            query_vector = self.encoder.encode([f"query: {query}"]).astype("float32")
        scores, indices = self.dense_index.search(query_vector, k)

        return [{"content": self.corpus[idx], "score": float(score)} for score, idx in zip(scores[0], indices[0], strict=False) if idx < len(self.corpus)]


# Flask app
app = Flask(__name__)
retriever = None


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "corpus_size": len(retriever.corpus), "index_type": "dense_only", "index_loaded": retriever.dense_index is not None})


@app.route("/retrieve", methods=["POST"])
def retrieve():
    """Main retrieval endpoint."""
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request"}), 400

        query = data["query"]
        k = data.get("top_k", data.get("k", 10))

        results = retriever.search(query=query, k=k)

        formatted_results = [{"id": f"doc_{i}", "content": result["content"], "score": result["score"]} for i, result in enumerate(results, 1)]

        return jsonify({"query": query, "method": "dense", "results": formatted_results, "num_results": len(formatted_results)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Dense-only retrieval server")
    parser.add_argument("--data_dir", default=DEFAULT_INDEX_DIR, help="Directory containing corpus and dense index")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--device", default="cpu", help="Device for E5 encoder (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--gpu_memory_limit_mb", type=int, default=1024, help="Max GPU memory in MB for E5 encoder (default: 1024)")

    args = parser.parse_args()

    # Initialize retriever
    global retriever
    try:
        retriever = LocalRetriever(args.data_dir, device=args.device, gpu_memory_limit_mb=args.gpu_memory_limit_mb)
        print(f"Dense retrieval server initialized with {len(retriever.corpus)} documents")
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        return

    # Start server
    print(f"Starting dense retrieval server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
