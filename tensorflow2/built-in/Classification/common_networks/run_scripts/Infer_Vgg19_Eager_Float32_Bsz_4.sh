#!/bin/bash
set -ex

function check_or_create_dir(){
    dst_dir="${1}"
    if [ ! -d "${dst_dir}" ]; then
        mkdir -p "${dst_dir}"
    fi
}

function parameter_validate(){
    run_eagerly=${1}
    quant_precision=${2}
    if [ $# -ne 2 ]; then
        echo "cambricon-error:expect parameter number to be 2, now it is $#"
        return 1
    fi
    if [ ${run_eagerly} -eq 1 ]; then
        if [ "${quant_precision}" != "fp32" ] && [ "${quant_precision}" != "fp16" ];then
            echo "cambricon-error:eager mode only support fp32/fp16 precision,now is ${quant_precision}"
            return 1
        fi
    fi
    return 0
}

while getopts ":e:b:p:" opt
do
    case $opt in
     e)
         eager_mode=${OPTARG}
     ;;
     b)
         input_bsz=${OPTARG}
     ;;
     p)
         input_precision=${OPTARG}
     ;;
     ?)
         echo "Unknow argument ${opt}"
         exit 1
     ;;
     esac
done

cur_path=$(pwd)
# begin:
# modify the below parameters according to your own demands
export PYTHONPATH=$cur_path:${PYTHONPATH}
source ../env.sh

run_eagerly=1
batch_size=4
quant_precision="fp32"

if [ -n "${eager_mode}" ];then
    run_eagerly="${eager_mode}"
fi

if [ -n "${input_bsz}" ];then
    batch_size="${input_bsz}"
fi

if [ -n "${input_precision}" ];then
    quant_precision="${input_precision}"
fi
parameter_validate ${run_eagerly} ${quant_precision}
if [ $? -ne 0 ];then
    exit 1
fi
# end

net_name="VGG19"
pub_model_path="${cur_path}/../pub_model_path"
native_savedmodel_dir="/data/tensorflow/training/models/tf2mm/saved_model/VGG19"
converted_savedmodel_dir="${pub_model_path}/converted_savedmodel_dir"
work_dir="${cur_path}/../"
result_path=${work_dir}/output_dir/result
base_path=$(dirname ${result_path})

check_or_create_dir "${native_savedmodel_dir}"
check_or_create_dir "${converted_savedmodel_dir}"
check_or_create_dir "${base_path}"

pushd "${work_dir}"
python classifier_infer.py \
        --model="${net_name}" \
        --pretrained_filepath="${native_savedmodel_dir}" \
        --model_dir="${converted_savedmodel_dir}" \
        --run_eagerly=${run_eagerly} \
        --data_dir="${DATA_DIR}" \
        --result_path="${result_path}" \
        --quant_precision="${quant_precision}" \
        --batch_size=$batch_size
popd
