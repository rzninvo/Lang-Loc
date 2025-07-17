import yaml
import os
import json

def load_config(config_path="config/default.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def get_download_config():
    config = load_config()
    base_dir = config['paths']['base_data_dir']
    label_map = os.path.join(base_dir, config['download']['label_map_filename'])
    files = config['download']['file_types']
    return base_dir, label_map, files

if __name__ == "__main__":
    try:
        base_dir, label_map, files = get_download_config()
        print(json.dumps({
            "base_dir": base_dir,
            "label_map": label_map,
            "file_types": files
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        exit(1)