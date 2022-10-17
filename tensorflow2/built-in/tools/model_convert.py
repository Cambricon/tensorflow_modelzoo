import tensorflow as tf
import sys
import os
import pprint
from tensorflow.python.compiler.magicmind import mm_convert as mm
from tensorflow.python.tools import saved_model_utils
import numpy as np
import copy
import uuid
from absl import logging
from saved_model_utils import *

def load_mm_converter(flags_obj, input_infos=None):
    """Loads a saved model using a TF-MM converter, and returns the converter"""
    precision = flags_obj.quant_precision
    params = copy.deepcopy(mm.DEFAULT_MM_CONVERSION_PARAMS)
    use_calibration = False
    if precision == "int8_fp16":
        precision_mode = mm.MMPrecisionMode.QINT8_MIXED_FLOAT16
        use_calibration = True
    elif precision == "int8_fp32":
        precision_mode = mm.MMPrecisionMode.QINT8_MIXED_FLOAT32
        use_calibration = True
    elif precision == "int16_fp16":
        precision_mode = mm.MMPrecisionMode.QINT16_MIXED_FLOAT16
        use_calibration = True
    elif precision == "int16_fp32":
        precision_mode = mm.MMPrecisionMode.QINT16_MIXED_FLOAT32
        use_calibration = True
    elif precision == "fp16":
        precision_mode = mm.MMPrecisionMode.FORCE_FLOAT16
    elif precision == "fp32":
        precision_mode = mm.MMPrecisionMode.FORCE_FLOAT32
    else:
        raise ValueError("{} is not supported for now.".format(precision))

    quantize_algos = {
        "linear": mm.MMQuantizationAlgorithm.LINEAR_ALGORITHM,
        "eqnm": mm.MMQuantizationAlgorithm.EQNM_ALGORITHM,
    }
    quantize_types = {
        "symm": mm.MMQuantizationType.symmetric,
        "asym": mm.MMQuantizationType.asymmetric,
    }
    quantize_granularitys = {
        "per_tensor": mm.MMQuantizationGranularity.per_tensor,
        "per_axis": mm.MMQuantizationGranularity.per_axis,
    }

    # parse optimze config
    opt_str = flags_obj.opt_config
    opt_configs = []
    if opt_str != None:
        opt_list = [
            item.strip() for item in opt_str.split(",") if len(item.strip()) != 0
        ]
        # opt_configs = []
        for item in opt_list:
            if item.lower() == "type64to32_conversion":
                opt_configs.append(mm.MMOptimizeConfig.TYPE64TO32_CONVERSION)
            elif item.lower() == "conv_scale_fold":
                opt_configs.append(mm.MMOptimizeConfig.CONV_SCALE_FOLD)
            else:
                raise ValueError("opt config:{} is not supported for now.".format(item))

    if use_calibration:
        if (
            (flags_obj.quantize_algo not in quantize_algos.keys())
            or (flags_obj.quantize_type not in quantize_types.keys())
            or (flags_obj.quantize_granularity not in quantize_granularitys.keys())
        ):
            raise ValueError(
                "quantize params error! please check:"
                "quantize_type:{type_val}\t support:{type_keys}\n"
                "quantize_algo:{algo_val}\t support:{algo_keys} \n"
                "quantize_granularity:{gran_val} support:{gran_keys}".format(
                    type_val=flags_obj.quantize_type,
                    type_keys=quantize_types.keys(),
                    algo_val=flags_obj.quantize_algo,
                    algo_keys=quantize_algos.keys(),
                    gran_val=flags_obj.quantize_granularity,
                    gran_keys=quantize_granularitys.keys(),
                )
            )
        params = params._replace(
            precision_mode=precision_mode,
            use_calibration=True,
            quantization_alg=quantize_algos[flags_obj.quantize_algo],
            quantization_type=quantize_types[flags_obj.quantize_type],
            quantization_granularity=quantize_granularitys[
                flags_obj.quantize_granularity
            ],
        )
    else:
        params = params._replace(
            precision_mode=precision_mode,
        )
    tmp_folder = os.path.join(TMP_FOLDER,str(uuid.uuid4()))
    target_model_dir = flags_obj.pretrained_filepath
    input_shapes = get_model_inputoutput_shape(target_model_dir)["inputs"].values()
    if flags_obj.enable_dim_range==False and has_dynamic_dim(input_shapes) and input_infos != None:
        input_shapes={}
        for key in input_infos.keys():
            input_shapes[key]=copy.copy(input_infos[key]["shape"])
            if input_shapes[key][0] == -1:
                input_shapes[key][0]=flags_obj.batch_size
        target_model_dir = savedmodel_fix_dynamicmodel(target_model_dir, tmp_folder, input_shapes)
    if flags_obj.enable_dim_range==False and has_dynamic_dim(input_shapes) and flags_obj.model != "Transformer":
        #need to fix batchsize dim
        target_model_dir = savedmodel_dynamicbatch_to_fixbatch(target_model_dir, tmp_folder, flags_obj.batch_size)

    if bool(flags_obj.enable_dim_range) == True:
        params = params._replace(
            dim_range_gen_strategy=mm.MMDimRangeGenStrategy.RANGE,
        )
    if len(opt_configs) != 0:
        params = params._replace(optimize_configs=opt_configs)
    params = params._replace(minimum_segment_size=flags_obj.minimum_segment_size)
    print("%" * 85)
    pprint.pprint(params)
    print("%" * 85)

    converter = mm.MMGraphConverterV2(
        input_saved_model_dir=target_model_dir, conversion_params=params
    )

    return converter


def model_convert(flags_obj, res_savedmodel_dir, input_infos=None):
    print("..........getting model input output shapes..........")
    native_savedmodel_dir = flags_obj.pretrained_filepath

    modelinfo = get_model_inputoutput_shape(native_savedmodel_dir)

    print("%" * 85)
    pprint.pprint(modelinfo)
    print("%" * 85)
    print("..........loading MM converter..........")

    converter = load_mm_converter(flags_obj, input_infos)

    input_shape = list(modelinfo["inputs"].values())[0]

    if input_shape[0] == -1:
        batch_size = flags_obj.batch_size
    else:
        batch_size = input_shape[0]

    def calibrate_fn():
        if flags_obj.calibration_data_dir == None:
            data_dir = flags_obj.data_dir
            logging.warning(
                "Warning:calibration_data_dir is None, use data_dir instead."
            )
        else:
            data_dir = flags_obj.calibration_data_dir
        dataset = Dataset(data_dir, batch_size=batch_size)
        data_dict = dataset.get_batch()
        while data_dict is not None:
            yield data_dict["tensor"]
            data_dict = dataset.get_batch()

    def dim_range_gen_fn():
        inputs_shape = list(modelinfo["inputs"].values())
        for insert_batch_size in [1, 2 * batch_size]:
            target_input = []
            for i in range(len(inputs_shape)):
                target_shape = inputs_shape[i]
                target_shape[0] = insert_batch_size
                target_input.append(
                    np.random.random_sample(target_shape).astype(np.float32)
                )

            yield target_input

    print("..........starting convert..........")
    enable_dim_range = flags_obj.enable_dim_range
    if converter._need_calibration and (enable_dim_range == False):
        print("-----convert with calibrate-----")
        converter.convert(calibration_input_fn=calibrate_fn)
    elif (converter._need_calibration == False) and enable_dim_range:
        print("-----convert with dim range generator-----")
        converter.convert(dim_range_input_fn=dim_range_gen_fn)
    elif converter._need_calibration and enable_dim_range:
        print("-----convert with calibrate and dim range generator-----")
        converter.convert(
            calibration_input_fn=calibrate_fn, dim_range_input_fn=dim_range_gen_fn
        )
    else:
        converter.convert()

    print("..........saving converted model..........")
    converter.save(res_savedmodel_dir)
