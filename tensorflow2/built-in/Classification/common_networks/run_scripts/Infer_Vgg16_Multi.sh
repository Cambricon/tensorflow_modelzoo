#!/bin/bash
set -x
run_eagerly=0
for bsz in 1 4;do
    # eager mode
    for quant_precision in fp16 fp32;do
            bash Infer_Vgg16_Eager_Float32_Bsz_4.sh -e 1 -b $bsz -p $quant_precision
    done
    # jit mode
    for quant_precision in fp16 fp32;do
            bash Infer_Vgg16_Jit_Float16_Bsz_4.sh -e $run_eagerly -b $bsz -p $quant_precision
    done

done


