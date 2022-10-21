#!/bin/bash

function print_usage() {
  RED='\033[0;31m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NONE='\033[0m'

  echo -e "${BOLD}bash ci_pipeline.sh${NONE} Command [Options]"

  echo -e "\n${RED}Command${NONE}:
    ${BLUE}precheckin_tf2${NONE}: [resnet50_cmcc] default to run all mdoels in cases/precheckin_tf2.json
    ${BLUE}usage${NONE}: display this message
  "
}


function precheckin_tf2() {
  python ${CI_PROJECT_DIR}/ci/precheckin/main.py \
    --test_json_file ${CI_PROJECT_DIR}/ci/cases/precheckin_tf2.json \
    --model_name  ""${@}
}


function main() {
  if [ -z $CI_PROJECT_DIR ]; then
    cd $( dirname ${BASH_SOURCE} )/..
    export CI_PROJECT_DIR=$(pwd)
    cd - > /dev/null
    echo ">>>>>>>>>>>>>>>>>>> CI_PROJECT_DIR is " $CI_PROJECT_DIR
  fi

  pip install pytest --quiet
  local cmd=$1
  case $cmd in
    precheckin_tf1)
      precheckin_tf1
      ;;
    precheckin_tf2)
      precheckin_tf2 ${@:2}
      ;;
    usage)
      print_usage
      ;;
    *)
      print_usage
      ;;
  esac
}


main $@
