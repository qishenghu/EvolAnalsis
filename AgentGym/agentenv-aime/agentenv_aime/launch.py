"""
Entrypoint for the aime agent environment.
"""

import argparse

import uvicorn

from .utils import debug_flg


def launch():
    """entrypoint for `aime` command"""

    parser = argparse.ArgumentParser()
    # Server configuration
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--workers", type=int, default=1)
    
    args = parser.parse_args()
    
    uvicorn.run(
        "agentenv_aime:app",
        host=args.host,
        port=args.port,
        reload=debug_flg,
        workers=args.workers,
    )
