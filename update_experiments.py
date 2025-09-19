import json
import argparse
from pathlib import Path

def update_experiments_config(config_path, updates, reset=False):
    config_path = Path(config_path)
    if not config_path.is_file():
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    if reset:
        experiments_dict = {}
    else:
        experiments_dict = config.setdefault('experiments', {})

    experiments_dict.update(updates)

    print("--- Current Configuration ---")
    print(json.dumps(config, indent=2))
    print("-----------------------------")

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Updates key-value pairs within the 'experiments' field of a config.json file."
    )
    parser.add_argument("config_path", type=str, help="Path to the config.json file.")
    parser.add_argument("--set", nargs=2, action='append', metavar=('KEY', 'VALUE'),
                        help="Set a key-value pair in the 'experiments' dict. Can be used multiple times. "
                             "e.g., --set learning_rate 0.001 --set batch_size 32")
    parser.add_argument("--reset", action='store_true',
                        help="Reset the 'experiments' dictionary before applying any updates.")

    args = parser.parse_args()

    if not args.set:
        args.set = {}

    updates_to_make = {}
    for key, value in args.set:
        try:
            processed_value = int(value)
        except ValueError:
            try:
                processed_value = float(value)
            except ValueError:
                if value.lower() == 'true':
                    processed_value = True
                elif value.lower() == 'false':
                    processed_value = False
                else:
                    processed_value = value
        updates_to_make[key] = processed_value

    update_experiments_config(args.config_path, updates_to_make, reset=args.reset)
