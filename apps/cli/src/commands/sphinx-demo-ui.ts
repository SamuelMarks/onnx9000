import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";

export async function handleSphinxDemoUICommand(args: string[]) {
  if (args.includes("-h") || args.includes("--help")) {
    console.log(
      "Usage: onnx9000 sphinx-demo-ui [options] \n\nStart the Sphinx Demo UI application. \n    ",
    );
    process.exit(0);
    return;
  }

  console.log("Starting Sphinx Demo UI...");

  // Find the workspace root
  let currentDir = process.cwd();
  let uiDir = "";

  while (currentDir !== "/") {
    if (fs.existsSync(path.join(currentDir, "pnpm-workspace.yaml"))) {
      uiDir = path.join(currentDir, "apps/sphinx-demo-ui");
      break;
    }
    currentDir = path.dirname(currentDir);
  }

  if (!uiDir || !fs.existsSync(uiDir)) {
    console.error("Sphinx Demo UI directory not found.");
    process.exit(1);
    return;
  }

  return new Promise<void>((resolve, reject) => {
    const child = spawn("pnpm", ["dev"], {
      cwd: uiDir,
      stdio: "inherit",
    });

    process.on("SIGINT", () => {
      child.kill("SIGINT");
      resolve();
    });

    child.on("error", (err) => {
      reject(err);
    });

    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Sphinx Demo UI exited with code ${code}`));
      }
    });
  });
}
