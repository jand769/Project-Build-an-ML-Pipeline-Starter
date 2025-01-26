import subprocess

# List of environments to remove
envs_to_remove = [
    "mlflow-0101fa0b8f222efdded2fff869b15ae3e45ca4ff",
    "mlflow-070f2cd3d506ab51cfd9698829becef5f98873be",
    "mlflow-7abf8355a2ba704267574dc7816c15f711f91c80",
    "mlflow-8287cf40e278416f8442ae395106792ced2fce02",
    "mlflow-a629edebd92bb6f50cb67a2df4a30dacbfd6ad64",
    "mlflow-b0fead796c6969ff73fa7b728c7fabe47e620afd",
    "mlflow-c55f48d1475615deffd8852f414384708595655c",
    "mlflow-c7f13156e34fd43f0e1fe87a443e20ed381ea872",
    "mlflow-fa2222139669491616f7a3cf4330b740aba5a8f7",
    "mlflow-ff2febd73ca20063054170516f787be55f755a07",
    "nyc_airbnb_dev"  # Add your current environment to remove
]

# Remove each environment
for env in envs_to_remove:
    print(f"Removing environment: {env}")
    try:
        subprocess.run(["conda", "env", "remove", "-n", env], check=True)
        print(f"Environment {env} removed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to remove environment {env}. Error: {e}")
