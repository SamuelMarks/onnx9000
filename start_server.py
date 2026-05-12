import os
import subprocess
import sys
import time

env = os.environ.copy()
env["PW_TEST_WEBSERVER_URL"] = "http://localhost:8000"

p = subprocess.Popen(["uv", "run", "python", "apps/cli/src/onnx9000_cli/main.py", "serve"])
time.sleep(2)

subprocess.run(
    ["pnpm", "playwright", "test", "e2e/demo-tvm.spec.ts", "--project=chromium"], env=env
)
subprocess.run(
    ["pnpm", "playwright", "test", "e2e/demo-tensorrt.spec.ts", "--project=chromium"], env=env
)
subprocess.run(
    ["pnpm", "playwright", "test", "e2e/demo-diffusers.spec.ts", "--project=chromium"], env=env
)

p.kill()
