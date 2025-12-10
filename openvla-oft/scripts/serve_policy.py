#!/usr/bin/env python
"""Launch a websocket server for OpenVLA(-OFT) policies."""

from experiments.robot.libero.policy_server import serve_policy


if __name__ == "__main__":
    serve_policy()

