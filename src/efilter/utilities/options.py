import argparse
import sys
from pathlib import Path

from eFilter.src.efilter.utilities.logging import log


# Arg     Explanation
# ---     ---------------------------------------
# -i      input file path(s)
# -l      set log level
# -d      log level: debug

INPUT_ARG = "input"
LOGGING_ARG = "loglevel"


class Options:
    """Class for handling input command-line arguments to eFilter"""

    def __init__(self):
        self.INPUT_PATH: Path

        arg_env = self._parse_command_line_args()
        if arg_env is None:
            return

        self._interpret_args(arg_env)

    def is_runnable(self):
        """
        After parsing the input command-line or configuration file,
        do we have the minimum requirements to execute?
        """
        return any([self.INPUT_PATH, self.RECONSTRUCTION_FILE])
        # self.OUTPUT_PATH is not None or \

    def _parse_command_line_args(self):
        """
        Analyze the command-line arguments.
        A reconstrcution file is necessary for direct reconstruction.

        @output: argument environment created by argparse
        """

        # Add full help message on incorrect parameters
        class MyParser(argparse.ArgumentParser):
            def error(self, message):
                sys.stderr.write(f"error: {message}\n")
                self.print_help()
                sys.exit(2)

        parser = MyParser(description="eFilter")

        #
        # eFilter arguments
        #
        parser.add_argument("-" + INPUT_ARG[0], "--" + INPUT_ARG, type=str, help="Input path to molecules to analyze.", required=True)
        parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="Quick flag to set logging level to debug.")

        parser.add_argument(
            "-l",
            dest="logLevel",
            choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            # default = 'INFO',
            default="INFO",  # NOTE: Set to DEBUG until we decide to move to production
            type=str.upper,
            help="Set the logging level to print to console.",
        )

        args = parser.parse_args()

        if not args.debug:
            log.setLevel(args.logLevel)
        else:
            log.setLevel("DEBUG")

        if args.logLevel == "NOTSET":
            log.disabled = True

        return args

    def _interpret_args(self, arg_env) -> None:
        """Set the user-defined options"""
        for arg in vars(arg_env):
            log.debug(f"{(arg + ':').upper():<11}{vars(arg_env)[arg]}")

            if arg == INPUT_ARG:
                input_path = getattr(arg_env, arg)
                if not Path(input_path).exists():
                    log.error(f"Input path does not exist: {input_path}.")
                    raise FileNotFoundError(f"Input path does not exist: {input_path}.")

                self.INPUT_PATH = Path(input_path)

    def __str__(self):
        """
        Report the current preferences for each option available
        """
        return f"Input Path: {self.INPUT_PATH}\n\
                Output Path: {self.OUTPUT_PATH}\n\
                Reconstruction file: {self.RECONSTRUCTION_FILE}\n"
