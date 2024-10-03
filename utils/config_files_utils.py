import yaml


def read_yaml(yaml_file: str) -> dict:
    with open(yaml_file, 'r') as config_file:
        try:
            yaml_dict = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict