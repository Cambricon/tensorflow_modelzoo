#!/bin/bash
set -ex
run_eagerly=0
for bsz in 128;do
    # eager mode
    for quant_precision in fp16 fp32;do
            bash infer_run_eager_fp32_bsz_128.sh  -e 1 -b $bsz -p $quant_precision
    done
    # jit mode
    for quant_precision in fp16 fp32;do
            bash infer_run_jit_fp32_bsz_128.sh -e $run_eagerly -b $bsz -p $quant_precision
    done
done
