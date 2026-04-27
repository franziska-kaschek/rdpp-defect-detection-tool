import yaml

def load_yaml_config(path: str):
    """
    Load a YAML configuration file.

    Config files are expected in `configs/`.
    """
    with open(f"configs/{path}", "r") as config_file:
        return yaml.safe_load(config_file)
    
def override(cli_value, default_value):
    """Return CLI value if provided, otherwise config default."""
    return cli_value if cli_value is not None else default_value
