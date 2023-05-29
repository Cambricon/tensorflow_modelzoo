#!/bin/bash
#set -e

usage () {
    echo "Usage:"
    echo "${0}    "
    echo "              data_dir: the_path of data_dir"
    echo "${0} PATHTO/data_dir"
}

if [ $# -lt 1 ]; then
  echo -e "\033[33m WARNING: DataDir is not set! Please Check It.\033[0m"
  usage
  exit -1
fi
data_dir=$1
pid=-1
if [ $# -eq 2 ]; then
    pid=$2
fi

os_name=NULL
function checkOS() {
  os_version=$(cat /etc/os-release | awk -F '=' '{if ($1=="PRETTY_NAME") print $2}' | awk -F '"' '{print $2}')
  os_name=$(cat /etc/os-release | awk -F '=' '{if ($1=="NAME") print $2}' | awk -F '"' '{print $2}')
  if [[ "$os_name" == "Ubuntu" ]] || [[ "$os_name" == "CentOS Linux" ]] || [[ "$os_name"  == "Debian GNU/Linux" ]]
  then
    echo -e "\033[32m OS: $os_version\033[0m"
  else
    echo -e "\033[31m ERROR: Only Support Ubuntu CentOs and Debian. But Running $os_version On The Machine.\033[0m"
    exit 1
  fi
}

function checkCPUInfo() {
    cpu_model=$(cat /proc/cpuinfo | awk -F ':' '{if ($1 ~ "model name") print $2}' | uniq)
    cpu_physical_core_num=$(cat /proc/cpuinfo |grep "physical id"|sort|uniq | wc -l)
    processor_num=$(cat /proc/cpuinfo | grep "processor" | wc -l)
    echo -e "\033[32m$cpu_model\033[0m"
    echo -e "\033[32m CPU Physical Core Nums: $cpu_physical_core_num\033[0m"
    echo -e "\033[32m CPU Processor Nums: $processor_num\033[0m"
}

function FindCPUProcess() {
  info=$(top -b -o +%CPU | head -n 20 | awk '{if (NR>7 && NR<11 && $9>5) print $1, $9, $12}' | awk -v pid=$pid '{if ($3!="top" && $1!=pid) printf("PID: %8s, COMMAND: %10s, Please Kill It.\n ", $1, $3)}')
  if [ "$info" != "" ]
  then
    echo -e "\033[33m WARNING: $info \033[0m"
  else
    echo -e "\033[32m No Programs Occupied CPUs!\033[0m"
  fi
}

function FindMLUProcess() {
  line_num=$(cnmon | awk '{if($0 ~ "PID") print NR}')
  pid_info=$(cnmon | awk -v line=$line_num -v pid=$pid '{a=line;if(NR>a && NF == 8 && $4!=pid) printf(" The PID %s Running On MLU, Please Kill It.\n", $4)}')
  if [ "$pid_info" != "" ]
  then
    echo -e "\033[33m WARNING: $pid_info\033[0m"
  fi
}

function checkCPUPerfMode() {
  if [ ! -f "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" ]
  then
    echo -e "\033[33m WARNING: Can't get cpu performance mode.\033[0m"
  else
    performance_mode=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    if [ "$performance_mode" == "" ]
    then
      echo -e "\033[33m WARNING: Can't get cpu performance mode2.\033[0m"
    elif [ "$performance_mode" != "performance" ]
    then
      echo -e "\033[33m WARNING: The CPU $performance_mode Mode Enabled! Please check it.\033[0m"
    fi
  fi
}

function checkIrqBalance() {
  irqbalance_ins=$(which irqbalance)
  if [ "$irqbalance_ins" == "" ]
  then
    echo -e "\033[33m WARNING: Can't get irqbalance.\033[0m"
  elif [ "$os_name" == "Debian GNU/Linux" ]
  then
    irqbalance_status=$(service irqbalance status | grep "is running")
    if [ "$irqbalance_status" == "" ]
    then
      echo -e "\033[33m WARNING: The CPU irqbalance isn't running! Please check it.\033[0m"
    fi
  else
    irqbalance_status=$(service irqbalance status | grep "running")
    if [ "$irqbalance_status" == "" ]
    then
      echo -e "\033[33m WARNING: The CPU irqbalance isn't running! Please check it.\033[0m"
    fi
  fi
}

function checkDataDir() {
    data_mount=$(df ${data_dir} | awk 'NR==2{print}' | awk -F' ' '{if(NF == 6) print $6}')
    if [ "$data_mount" == "" ]
    then
      echo -e "\033[33m WARNING: ${data_dir} does not exist.\033[0m"
      return
    fi
    if [ "$data_mount" == "/" ]
    then
      echo -e "\033[32m ${data_dir} is on local.\033[0m"
    else
      echo -e "\033[33m WARNING: ${data_dir} isn't on local, it may degrade performance.\033[0m"
    fi
}

function checkMLULinkDisable() {
  max_card_num=$x8_card_num
  success=1
  index_end=`expr $max_card_num - 1`
  for i in `seq 0 $index_end`
  do
      bool_card_exit=$(cnmon -c $i | grep "not existed")
      if [ "$bool_card_exit" != "" ]
      then
        return
      fi
      bool_mlulink=$(cnmon mlulink -c $i -s | awk -F ':' '{if(NF == 2) print $2}' | grep "Disable")
      if [ "$bool_mlulink" != "" ]
      then
        echo -e "\033[33m WARNING: Card$i's MLUlink ports are disabled. Please check it\033[0m"
        success=0
      fi
  done
  if [ "$success" == "1" ]
  then
    echo -e "\033[32m MLUlink Success.\033[0m"
  fi
}

function checkMLULinkEnable() {
  max_card_num=$x8_card_num
  success=1
  index_end=`expr $max_card_num - 1`
  for i in `seq 0 $index_end`
  do
      bool_card_exit=$(cnmon -c $i | grep "not existed")
      if [ "$bool_card_exit" != "" ]
      then
        return
      fi
      bool_mlulink=$(cnmon mlulink -c $i -s | awk -F ':' '{if(NF == 2) print $2}' | grep "Enable")
      if [ "$bool_mlulink" == "" ]
      then
        echo -e "\033[33m WARNING: Card$i's MLUlink ports are disabled. Please check it\033[0m"
        success=0
      fi
  done
  if [ "$success" == "1" ]
  then
    echo -e "\033[32m MLUlink Success.\033[0m"
  fi
}

function check_mlu_config() {
  max_card_num=16
  index_end=`expr $max_card_num - 1`
  for card in `seq 0 $index_end`
  do
    bool_card_exist=$(cnmon -c $card | grep "MLU" | grep "On")
    if [ "$bool_card_exist" == "" ]
    then
      return
    fi
    bool_card_is_mlu370x8=$(cnmon -c $card | grep "MLU370-X8")
    if [ "$bool_card_is_mlu370x8" != "" ]
    then
      bool_mlulink=1
      ((x8_card_num += 1))
    fi
  done
}

checkOS
checkCPUInfo
checkCPUPerfMode
checkIrqBalance
checkDataDir
FindCPUProcess
FindMLUProcess
bool_mlulink=0
x8_card_num=0
check_mlu_config

if [ "$bool_mlulink" == "1" ]
then
    if (($x8_card_num%4==0))
    then
        checkMLULinkDisable
    else
        checkMLULinkEnable
    fi
fi
