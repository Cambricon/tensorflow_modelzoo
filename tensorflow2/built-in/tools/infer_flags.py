from absl import flags
#flags = flags.FLAGS

# cambricon-note begin:flags used for TFMM
flags.DEFINE_string(
      "model",
      default=None,
      help="model name",
)

flags.DEFINE_string(
      "pretrained_filepath",
      default=None,
      help="the pretrained model file path.",
)

flags.DEFINE_string(
      "model_dir",
      default=None,
      help="the converted savedmodel file path.",
)

flags.DEFINE_integer(
      "run_eagerly",
      default=1,
      help="the flag of model running mode, support jit/eager.",
)

flags.DEFINE_string(
      "data_dir",
      default=None,
      help="the dataset file path.",
)

flags.DEFINE_integer(
      "batch_size",
      default=1,
      help="model batch size.",
)

flags.DEFINE_string(
      "opt_config",
      default=None,
      help="the flag of optimize configs, if added more than one flag,split with comma.support conv_scale_fold,type64to32_conversion.",
)
flags.DEFINE_string(
      "quantize_granularity",
      default="per_tensor",
      help="quantize granularity, support per_tensor/per_axis",
)

flags.DEFINE_string(
      "quantize_algo",
      default="linear",
      help="quantize algorithm, support linear/eqnm",
)

flags.DEFINE_string(
      "quantize_type", default="symm", help="quantize type, support symm/asym"
)
flags.DEFINE_integer(
        "warmup_count",
        default=1,
        help="Warmup step of the inference before offical inference",
)
flags.DEFINE_string(
      "calibration_data_dir",
      default=None,
      help="The folder path of the input image(*.jpg) data.",
)

flags.DEFINE_string(
      "imagenet_label_file", default=None, help="The label file for imagenet dataset."
)
flags.DEFINE_integer(
      "enable_dim_range", default=0, help="Enable dim range if the value >= 1."
)
flags.DEFINE_integer(
      "minimum_segment_size", default=3, help="minimum segment size, value >= 1."
)
flags.DEFINE_string("result_path", default="", help="Path for inference result.")
flags.DEFINE_integer(
      "top_k", default=5, help="top_k result displayed after inference."
)
flags.DEFINE_string(
      "quant_precision",
      default="fp32",
      help="Precision, optional val int8_fp16/int8_fp32/int16_fp16/int16_fp16/fp16/fp32,if the precision is int8_fp16/int8_fp32/int16_fp16/int16_fp32,converter will do calibration to the native savedmodel.",
)
# cambricon-note end:flags used for TFMM
