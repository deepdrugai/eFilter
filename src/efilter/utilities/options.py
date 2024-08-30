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
        parser.add_argument("-" + INPUT_ARG, type=str, help="Input path to molecules to analyze.", required=True)

        parser.add_argument("-d", dest="debug", action="store_true", help="Quick flag to set logging level to debug.")

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

        # CTA: Level-Based Reconstruction 1/26/2024
        # Type of reconstruction to perform
        # if args.direct:
        #     self.DIRECT = True
        # if args.level_based:
        #     self.LEVEL_BASED = True

        return args

    def _interpret_args(self, arg_env) -> None:
        """Set the user-defined options"""
        for arg in vars(arg_env):
            log.debug(f"{(arg + ':').upper():<11}{vars(arg_env)[arg]}")

            if arg == INPUT_ARG:
                input_path = getattr(arg_env, arg)
                if Path(input_path).exists():
                    if not Path(input_path).is_dir():
                        log.warning(f"Input path {input_path} is not a directory. Did you mean: {Path(input_path).parent}?")
                        input_path = Path(input_path).parent
                else:
                    log.error(f"Input path does not exist: {input_path}.")
                    raise FileNotFoundError(f"Input path does not exist: {input_path}.")

                self.INPUT_PATH = Path(input_path)

            elif arg == OUTPUT_ARG:
                self.OUTPUT_PATH = getattr(arg_env, arg)

            elif arg == RECONSTRUCTION_FILE_ARG:
                self.RECONSTRUCTION_FILE = getattr(arg_env, arg)
                if not self.RECONSTRUCTION_FILE:
                    log.debug("Reconstruction file not provided.")

                    # If the reconstruction file is None, search for it in the input directory
                    log.debug(f"Searching for reconstruction file in input ({self.INPUT_PATH}) directory...")
                    t_list = list(self.INPUT_PATH.glob("t-*"))
                    # log.debug(f"Reconstruction file{'s'[:len(t_list)^1]} found: {','.join(str(p) for p in t_list)}.")

                    # If no trace file or multiple trace files found, log error
                    if len(t_list) != 1:
                        log.error(f"Error identifying reconstruction file.")
                        return

                    # Set the reconstruction file to the first and only file found
                    self.RECONSTRUCTION_FILE = t_list[0]

                self.RECONSTRUCTION_FILE = Path(self.RECONSTRUCTION_FILE)
                log.info(f"Reconstruction file found: {self.RECONSTRUCTION_FILE}")

    def __str__(self):
        """
        Report the current preferences for each option available
        """
        return f"Input Path: {self.INPUT_PATH}\n\
                Output Path: {self.OUTPUT_PATH}\n\
                Reconstruction file: {self.RECONSTRUCTION_FILE}\n"
