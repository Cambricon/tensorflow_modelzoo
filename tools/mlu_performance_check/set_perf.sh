#!/bin/bash
#set -e

OS_NAME=NULL
function checkOS() {
  os_version=$(cat /etc/os-release | awk -F '=' '{if ($1=="PRETTY_NAME") print $2}' | awk -F '"' '{print $2}')
  OS_NAME=$(cat /etc/os-release | awk -F '=' '{if ($1=="NAME") print $2}' | awk -F '"' '{print $2}')
  if [[ "$OS_NAME" == "Ubuntu" ]] || [[ "$OS_NAME" == "CentOS Linux" ]] || [[ "$OS_NAME"  == "Debian GNU/Linux" ]]
  then
    echo -e "\033[32m OS: $os_version\033[0m"
  else
    echo -e "\033[31m ERROR: Only Support Ubuntu/CentOs and Debian. But Running $os_version On The Machine.\033[0m"
    exit 1
  fi
}

function setCPUPerfMode() {
  if [ "$OS_NAME" == "Ubuntu" ]
  then
    installed_version=$(dpkg -l linux-tools-$(uname -r) | grep linux-tools-$(uname -r) | awk '{print $3}')
    sys_version=$(uname -r | awk -F '-generic' '{print $1}')
    bool_match=$(echo $installed_version | grep $sys_version)
    if [ "$bool_match" == "" ]
    then
      apt-get install -y linux-tools-$(uname -r)
    fi
  elif [ "$OS_NAME" == "CentOS Linux" ]
  then
	installed_version=$(cpupower -v | awk '{if(NR==1) print $2}' | awk -F '.debug' '{print $1}')
	sys_version=$(uname -r | awk -F '-generic' '{print $1}')
	bool_match=$(echo $installed_version | grep $sys_version)
	if [ "$bool_match" == "" ]
	then
	  yum install -y cpupowerutils
	fi
  elif [ "$OS_NAME" == "Debian GNU/Linux" ]
  then
	bool_install=$(dpkg -l linux-cpupower | grep "Version")
	if [ "$bool_install" == "" ]
	then
	  apt-get install -y linux-cpupower
	fi
  else
    echo -e "\033[31m ERROR: Set Performance Mode Failed. Only Support Ubuntu/CentOs and Debian. But Running $os_version On The Machine.\033[0m"
    exit 1
  fi
  performance_mode=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
  if [ "$performance_mode" != "performance" ]
  then
    perf_cpu=$(cpupower -c all frequency-set -g performance)
    echo -e "\033[32m $perf_cpu \033[0m"
    # check performance mode
    performance_mode=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    if [ "$performance_mode" == "performance" ]
    then
      echo -e "\033[32m The CPU Performance Mode Enabled!\033[0m"
    else
      echo -e "\033[31m ERROR: The CPU $performance_mode Mode Enabled! Please Check It.\033[0m"
      exit 1
    fi
  else
    echo -e "\033[32m The CPU $performance_mode Mode Enabled!\033[0m"
  fi
}

function startIrqBalance() {
  irqbalance_ins=$(which irqbalance)
  if [ "$irqbalance_ins" == "" ]
  then
    if [ "$OS_NAME" == "CentOS Linux" ]
    then
	  yum install -y irqbalance
    else
	  apt-get install -y irqbalance
    fi
  elif [ "$OS_NAME" == "Debian GNU/Linux" ]
  then
    irqbalance_status=$(service irqbalance status | grep "is running")
    if [ "$irqbalance_status" == "" ]
    then
      $(service irqbalance start)
      irqbalance_status=$(service irqbalance status | grep "is running")
      if [ "$irqbalance_status" == "" ]
      then
        echo -e "\033[31m The CPU irqbalance isn't running! Please Check It.\033[0m"
        exit 1
      fi
    fi
  else
    irqbalance_status=$(service irqbalance status | grep "running")
    if [ "$irqbalance_status" == "" ]
    then
      $(service irqbalance start)
      irqbalance_status=$(service irqbalance status | grep "running")
      if [ "$irqbalance_status" == "" ]
      then
        echo -e "\033[31m ERROR: The CPU irqbalance isn't running! Please Check It.\033[0m"
        exit 1
      fi
    fi
  fi
  echo -e "\033[32m The CPU irqbalance is running!\033[0m"
}

checkOS
setCPUPerfMode
startIrqBalance
