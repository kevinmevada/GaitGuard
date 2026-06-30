#!/usr/bin/env python
"""Entry point: python hpc.py manifests ingest | shard ingest --manifest ... | merge ingest"""

from src.hpc.cli import main

if __name__ == "__main__":
    main()
