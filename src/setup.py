import os
import json

def set_environment_variables():
    # Define the path to the configurations file
    config_file_path = os.path.join(os.path.dirname(__file__), '..', 'configurations.json')

    # Read and parse the JSON file
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    # Extract configuration values
    gradient_access_token = config.get('access_token')
    gradient_workspace_id = config.get('workspace_id')

    # Check if environment variables are already set
    if 'GRADIENT_ACCESS_TOKEN' not in os.environ:
        os.environ['GRADIENT_ACCESS_TOKEN'] = gradient_access_token
        print("GRADIENT_ACCESS_TOKEN set.")

    if 'GRADIENT_WORKSPACE_ID' not in os.environ:
        os.environ['GRADIENT_WORKSPACE_ID'] = gradient_workspace_id
        print("GRADIENT_WORKSPACE_ID set.")

if __name__ == "__main__":
    set_environment_variables()
