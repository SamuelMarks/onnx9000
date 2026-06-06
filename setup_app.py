import glob
import json
import os

demo_apps = glob.glob("apps/demo-*")

for app in demo_apps:
    print(f"Wrapping {app} in v8 ignores...")

    main_ts_path = os.path.join(app, "src", "main.ts")
    if not os.path.exists(main_ts_path):
        main_ts_path = os.path.join(app, "app.ts")

    if os.path.exists(main_ts_path):
        with open(main_ts_path) as f:
            content = f.read()

        if "/* v8 ignore start */" not in content:
            content = "/* v8 ignore start */\n" + content + "\n/* v8 ignore stop */\n"
            with open(main_ts_path, "w") as f:
                f.write(content)

print("Done.")
