import json
import os
import subprocess
import sys


KIT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(KIT_DIR)
CONFIG_PATH = os.path.join(KIT_DIR, "configs", "industrial_config.json")


def assert_file(path):
    if not os.path.exists(path):
        raise AssertionError(f"missing file: {path}")


def main():
    required_files = [
        "README.md",
        "docs/algorithm_selection.md",
        "docs/deployment_playbook.md",
        "configs/industrial_config.json",
        "src/edge_runtime.py",
        "src/solution_matrix.py",
    ]
    for item in required_files:
        assert_file(os.path.join(KIT_DIR, item))

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    model_path = os.path.abspath(os.path.join(os.path.dirname(CONFIG_PATH), config["model_path"]))
    source_path = os.path.abspath(os.path.join(os.path.dirname(CONFIG_PATH), config["source"]))
    assert_file(model_path)
    assert_file(source_path)

    matrix = subprocess.run(
        [sys.executable, os.path.join(KIT_DIR, "src", "solution_matrix.py")],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    parsed_matrix = json.loads(matrix.stdout)
    if len(parsed_matrix) < 5:
        raise AssertionError("solution matrix is incomplete")

    runtime = subprocess.run(
        [
            sys.executable,
            os.path.join(KIT_DIR, "src", "edge_runtime.py"),
            "--config",
            CONFIG_PATH,
            "--max-frames",
            "2",
        ],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(runtime.stdout)
    if summary["frames"] != 2:
        raise AssertionError(f"unexpected frame count: {summary}")
    assert_file(summary["event_path"])
    assert_file(summary["video_path"])
    print("✅ industrial delivery kit validation passed")


if __name__ == "__main__":
    main()
