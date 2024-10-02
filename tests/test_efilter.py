import glob
import re
import subprocess

import pytest

from efilter.utilities.logging import log


@pytest.mark.parametrize("test_file", glob.glob("data/DB*"))
def test_efilter(test_file):
    # Run efilter command for multiple example files
    # TODO: Add -d flag to run efilter in debug mode
    result = subprocess.run(["efilter", "-i", test_file], capture_output=True, text=True, check=False)

    check_efilter_logfile(test_file, result)

    # Check if efilter ran successfully
    log.debug(f"Result Return Code: {result.returncode}")
    assert result.returncode == 0


@pytest.mark.parametrize("test_file", glob.glob("data/failed/DB*"))
def test_efilter_failing(test_file):
    # Run efilter command for multiple example files
    # TODO: Add -d flag to run efilter in debug mode
    result = subprocess.run(["efilter", "-i", test_file], capture_output=True, text=True, check=False)

    check_efilter_logfile(test_file, result)

    # Check if efilter ran successfully
    log.debug(f"Result Return Code: {result.returncode}")
    assert result.returncode == 1


def check_efilter_logfile(test_file, result):
    expected_output = test_file + "/expected_output.txt"

    with open(expected_output) as f:
        expected_output = f.readlines()

    # Check if efilter logs are same as expected
    # Use a regular expression to remove the line number
    # Use .split(" [") to remove the timestamp
    # stderr_output = [re.sub(r":\d+::", ":::", line.split(" [")[-1].strip()) for line in result.stderr.splitlines()]
    # expected_output = [re.sub(r":\d+::", ":::", line.split(" [")[-1].strip()) for line in expected_output]
    # lines_with_timestamp = []
    # lines_without_timestamp = []
    if result.stderr:
        print("result.stderr length: ", len(result.stderr.splitlines()))
        print(result.stderr.splitlines())
        # lines_with_timestamp = [line for line in result.stderr.splitlines() if " [" in line and "] - " in line]
        # lines_without_timestamp = [line for line in result.stderr.splitlines() if " [" not in line]

    if result.stdout:
        print("result.stdout length: ", len(result.stdout.splitlines()))
        print(result.stdout)

    stderr_output_logs = [line.split(" [")[-1].strip() for line in result.stderr.splitlines()]
    expected_output_logs = [line.split(" [")[-1].strip() for line in expected_output]

    for linea, lineb in zip(stderr_output_logs, expected_output_logs):
        print(f"Std Err:\t{linea}")
        print(f"Expected:\t{lineb}")
        print()
        # Use a regular expression to remove the line number from log line
        assert re.sub(r":\d+::", ":::", linea) == re.sub(r":\d+::", ":::", lineb)

    # if lines_without_timestamp:
    #     print("Non-efilter Std Errors:")
    #     for line in lines_without_timestamp:
    #         print(line)

    # assert stderr_output == expected_output
