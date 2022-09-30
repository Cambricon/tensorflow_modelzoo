import pytest
import sys
import subprocess


def test_precheckin(testcase):
    print("===== current test case =====")
    for cmd in testcase:
        print(cmd)
    print()

    command_line = " && ".join(testcase)
    run_args = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "shell": True,
        "bufsize": -1,
        "executable": "/bin/bash"
    }

    process = subprocess.Popen( [command_line], **run_args)
    msg, _ = process.communicate()
    msg = str(msg, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError("Error in run:\n {} \n".format("".join(command_line)) +
                           "msg: \n {}".format(msg))
    print(msg)


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([
            __file__, "-rP"] + sys.argv[1:]))
