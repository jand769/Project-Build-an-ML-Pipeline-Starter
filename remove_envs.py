import subprocess

# List of environments to remove
envs_to_remove = [
    "mlflow-070f2cd3d506ab51cfd9698829becef5f98873be",
    "mlflow-39827ffc24365ddbc14cbc317c51538dd071fbcc",
    "mlflow-7abf8355a2ba704267574dc7816c15f711f91c80",
    "mlflow-8e9a002676208cdf1bbfa1f0c57b48682a0c2196",
    "mlflow-93e44645c52df798bb67c8b7fe3cf075128243a1",
    "mlflow-b75b0eb9d2922d65e29812854458ae013f824787",
    "mlflow-c7f13156e34fd43f0e1fe87a443e20ed381ea872",
    "mlflow-f1798e912c7b842d8e1ae78e50bad6f5e4facf3a",
    "mlflow-fa2222139669491616f7a3cf4330b740aba5a8f7",
]

# Remove each environment
for env in envs_to_remove:
    print(f"Removing environment: {env}")
    try:
        subprocess.run(["conda", "env", "remove", "-n", env], check=True)
        print(f"Environment {env} removed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to remove environment {env}. Error: {e}")

