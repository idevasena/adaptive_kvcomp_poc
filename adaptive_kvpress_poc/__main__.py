"""Allow running the package directly: python -m adaptive_kvpress_poc [--fast]"""

from .run_simulation import main
import sys

if __name__ == "__main__":
    fast = "--fast" in sys.argv
    main(fast=fast)
