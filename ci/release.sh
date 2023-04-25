#!/bin/bash

if [[ ! ${TENSORFLOW_MODELZOO_HOME} ]];then
  if type git > /dev/null 2>&1 && git rev-parse --is-inside-work-tree > /dev/null 2>&1;then
    TENSORFLOW_MODELZOO_HOME=$( git rev-parse --show-toplevel )
  else
    1>&2 echo "ERROR: TENSORFLOW_MODELZOO_HOME is not set, please set TENSORFLOW_MODELZOO_HOME to tensorflow project root"
    exit 1
  fi
fi

# remove common
function remove_common(){
    rm -rf ${TENSORFLOW_MODELZOO_HOME}/.git*
    rm -rf ${TENSORFLOW_MODELZOO_HOME}/ci
    rm ${TENSORFLOW_MODELZOO_HOME}/release.sh
}

# remove tf1
function remove_tf1(){
    rm -rf ${TENSORFLOW_MODELZOO_HOME}/tensorflow/
}

# remove tf2
function remove_tf2(){
    rm -rf ${TENSORFLOW_MODELZOO_HOME}/tensorflow2/
}

# main
function main(){
    if [[ ! -n "$1" ]];then
        echo "remove common"
        remove_common
    elif [[ $1 == "tf1" ]];then
        echo "remove common"
        remove_common
        echo "remove tf2"
        remove_tf2
    elif [[ $1 == "tf2" ]];then
        echo "remove common"
        remove_common
        echo "remove tf1"
        remove_tf1
    else
        echo "Please input tf1 or tf2"
    fi
}
main $@
