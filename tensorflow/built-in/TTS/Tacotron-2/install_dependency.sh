#!/bin/bash
#set -e
os_version=$(cat /etc/os-release | awk -F '=' '{if ($1=="PRETTY_NAME") print $2}' | awk -F '"' '{print $2}')
os_name=$(cat /etc/os-release | awk -F '=' '{if ($1=="NAME") print $2}' | awk -F '"' '{print $2}')
if [[ "$os_name" == "Ubuntu" ]]
then
  apt-get update
  apt install -y python3-all-dev
  apt-get install -y portaudio19-dev
  apt-get install -y libsndfile1
elif [[ "$os_name" == "CentOS Linux" ]]
then
  yum -y install portaudio portaudio-devel
  yum -y install xz-devel
  yum -y install python-backports-lzma
  yum -y install libsndfile-devel
  pip install backports.lzma
else
  echo -e "\033[31m ERROR: Only Support Ubuntu Or CentOs. But Running $os_version On The Machine.\033[0m"
  exit 1
fi

array=(pyaudio sounddevice librosa==0.8.1 matplotlib unidecode inflect tqdm keras==2.3.1 numpy==1.21.6)

for element in ${array[@]}
do
   pip install $element -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
done
