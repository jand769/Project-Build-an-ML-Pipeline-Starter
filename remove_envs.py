import subprocess

# List of environments to remove
envs_to_remove = [
    "mlflow-070f2cd3d506ab51cfd9698829becef5f98873be",
    "mlflow-7abf8355a2ba704267574dc7816c15f711f91c80",
    "mlflow-8287cf40e278416f8442ae395106792ced2fce02",
    "mlflow-c7f13156e34fd43f0e1fe87a443e20ed381ea872",
    "nyc_airbnb_dev"  # Added your current environment to remove
]

# Remove each environment
for env in envs_to_remove:
    print(f"Removing environment: {env}")
    try:
        subprocess.run(["conda", "env", "remove", "-n", env], check=True)
        print(f"Environment {env} removed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to remove environment {env}. Error: {e}")
