import os
import sys

sys.path.append(os.path.abspath("."))
import scripts.generate_snapshots as gs
import scripts.update_badges as ub
import setup_app as sa
import setup_demo as sd
import start_server as ss


def test_scripts():
    assert gs.run() == "generate_snapshots"
    assert ub.run() == "update_badges"
    assert sa.run() == "setup_app"
    assert sd.run() == "setup_demo"
    assert ss.run() == "start_server"
