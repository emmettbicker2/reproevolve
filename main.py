from reproevolve.controller import Config, Controller
import yaml
import os
import argparse


def main(config_path: str, initial_program: str, evaluator_program: str):
    export_secrets_as_environment_variables()

    with open(config_path) as f:
        config_dict = yaml.safe_load(f.read())
        config = Config.model_validate(
            {
                **config_dict,
                "initial_program_path": initial_program,
                "evaluator_program_path": evaluator_program,
            }
        )

    controller = Controller(config)
    controller.run()


def export_secrets_as_environment_variables(secrets_path: str = "secrets.yaml"):
    with open(secrets_path) as f:
        secrets = yaml.safe_load(f.read())

    for k, v in secrets.items():
        os.environ[k] = v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main function for minevolve")
    parser.add_argument(
        "--initial-program", help="The path to the initial program", required=True
    )
    parser.add_argument(
        "--evaluator-program", help="The path to the evaluator program", required=True
    )
    args = parser.parse_args()
    main("config.yaml", args.initial_program, args.evaluator_program)
