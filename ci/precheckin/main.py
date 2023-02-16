import pytest
import sys
import os


def read_log(filename):
  bufsize = 2048
  file_size = os.path.getsize(filename)
  logFile = open(filename, 'rb')
  if file_size > bufsize:
    logFile.seek(0-bufsize, 2)
  return (b"".join(logFile.readlines())).decode('utf-8', 'ignore')


def test_precheckin(testcase):
    print("===== current test case =====")
    for cmd in testcase:
        print(cmd)
    print()

    run_cmd_path = "run_cmd.sh"
    with open(run_cmd_path, 'w') as f:
        f.write("#!/bin/bash\nset -ex\n")
        for cmd in testcase:
            f.write(f"{cmd}\n")
    
    ci_work_dir = os.environ.get('CI_WORK_DIR')
    log_path = f"{ci_work_dir}/tensorflow_modelzoo/test_log/run_cmd.log"
    with open("exec_cmd.sh", "w") as f:
        exec_cmd = f'''
            #!/bin/bash
            set -exo pipefail
            bash {run_cmd_path} 2>&1 | tee {log_path}
        '''
        f.write(exec_cmd)

    result = os.system("bash exec_cmd.sh")
    assert 0 == result, "\n[FAILED] log:{}".format(read_log(log_path))


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([
            __file__, "-sv"] + sys.argv[1:]))
