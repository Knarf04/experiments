import json
import shutil
import argparse
from pathlib import Path


def update_hf_config(config_path, updates, reset=False):
    config_path = Path(config_path)
    if not config_path.is_file():
        print(f"Error: Config file not found at {config_path}")
        return

    backup_path = config_path.with_name("_config.old")

    if reset and backup_path.is_file():
        # Restore from backup before applying new settings
        shutil.copy2(backup_path, config_path)
        print(f"Restored config from {backup_path}")

    # Back up original if no backup exists yet
    if not backup_path.is_file():
        shutil.copy2(config_path, backup_path)
        print(f"Backed up original config to {backup_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    config.update(updates)

    print("--- Current Configuration ---")
    print(json.dumps(config, indent=2))
    print("-----------------------------")

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Updates key-value pairs in a HuggingFace model config.json file."
    )
    parser.add_argument("config_path", type=str, help="Path to the config.json file.")
    parser.add_argument("--set", nargs=2, action='append', metavar=('KEY', 'VALUE'),
                        help="Set a key-value pair in the config. Can be used multiple times. "
                             "e.g., --set rope_theta 500000 --set max_position_embeddings 8192")
    parser.add_argument("--reset", action='store_true',
                        help="Restore from _config.old backup before applying any updates.")

    args = parser.parse_args()

    if not args.set:
        args.set = {}

    update_dict = {}
    for key, value in args.set:
        # v5.0.0dev specific rope format
        if "rope" in key:
            if "rope_parameters" not in update_dict:
                update_dict["rope_parameters"] = {}
            update_dict_entry = update_dict["rope_parameters"]
        else:
            update_dict_entry = update_dict

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

        update_dict_entry[key] = processed_value

    update_hf_config(args.config_path, update_dict, reset=args.reset)
