import os
import numpy as np
import tensorflow as tf


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class Evaluators(object):
    def __init__(self, args, configs=None) -> None:
        super().__init__()
        self.args = args
        self.configs = configs
        self.evaluators = self.GetEvaluators(args, configs)

    def GetEvaluators(self, args, configs):
        from object_detection import eval_util
        from object_detection.utils import label_map_util
        evaluator_options = eval_util.evaluator_options_from_eval_config(
            configs['eval_config'])
        if configs['eval_input_config'].label_map_path:
            class_aware_category_index = (label_map_util.create_category_index_from_labelmap(
                configs['eval_input_config'].label_map_path))
            class_aware_evaluators = eval_util.get_evaluators(configs['eval_config'],
                                                              list(class_aware_category_index.values()), evaluator_options)
            return class_aware_evaluators

    def FeedEvalData2Evaluators(self, eval_data):
        for evaluator in self.evaluators:
            evaluator.add_eval_dict(eval_data)

    def DoEvaluation(self):
        eval_metrics = {}
        for evaluator in self.evaluators:
            eval_metrics.update(evaluator.evaluate())
        # precison has been converted to be percent
        return {"map": eval_metrics["DetectionBoxes_Precision/mAP"] * 100}


class CenternetPostProcess(object):
    def __init__(self, args, out_stride=4, max_box_predictions=100,
                 output_map={"object_center": "output_2", "box_scale": "output_1",
                             "box_offset": "output_0"}) -> None:
        super(CenternetPostProcess, self).__init__()
        self.eval_dict = {}
        self.args = args
        self.MAX_BOX_PREDICTIONS = max_box_predictions
        self.STRIDE = out_stride
        self.output_map = output_map
        self.HASH_KEY = 'hash'

    def prepare_eval_dict(self, detections, groundtruth, features):
        """Prepares eval dictionary containing detections and groundtruth.

        Takes in `detections` from the model, `groundtruth` and `features` returned
        from the eval tf.data.dataset and creates a dictionary of tensors suitable
        for detection eval modules.

        Args:
            detections: A dictionary of tensors returned by `model.postprocess`.
            groundtruth: `inputs.eval_input` returns an eval dataset of (features,
            labels) tuple. `groundtruth` must be set to `labels`.
            Please note that:
                * fields.InputDataFields.groundtruth_classes must be 0-indexed and
                in its 1-hot representation.
                * fields.InputDataFields.groundtruth_verified_neg_classes must be
                0-indexed and in its multi-hot repesentation.
                * fields.InputDataFields.groundtruth_not_exhaustive_classes must be
                0-indexed and in its multi-hot repesentation.
                * fields.InputDataFields.groundtruth_labeled_classes must be
                0-indexed and in its multi-hot repesentation.
            features: `inputs.eval_input` returns an eval dataset of (features, labels)
            tuple. This argument must be set to a dictionary containing the following
            keys and their corresponding values from `features` --
                * fields.InputDataFields.image
                * fields.InputDataFields.original_image
                * fields.InputDataFields.original_image_spatial_shape
                * fields.InputDataFields.true_image_shape
                * inputs.HASH_KEY

        Returns:
            eval_dict: A dictionary of tensors to pass to eval module.
            class_agnostic: Whether to evaluate detection in class agnostic mode.
        """

        from object_detection.core import standard_fields as fields
        from object_detection.eval_util import result_dict_for_batched_example
        groundtruth_boxes = groundtruth[fields.InputDataFields.groundtruth_boxes]
        groundtruth_boxes_shape = tf.shape(groundtruth_boxes)
        # For class-agnostic models, groundtruth one-hot encodings collapse to all
        # ones.
        class_agnostic = (
            fields.DetectionResultFields.detection_classes not in detections)
        if class_agnostic:
            groundtruth_classes_one_hot = tf.ones(
                [groundtruth_boxes_shape[0], groundtruth_boxes_shape[1], 1])
        else:
            groundtruth_classes_one_hot = groundtruth[
                fields.InputDataFields.groundtruth_classes]
        label_id_offset = 1  # Applying label id offset (b/63711816)
        groundtruth_classes = (
            tf.argmax(groundtruth_classes_one_hot, axis=2) + label_id_offset)
        groundtruth[fields.InputDataFields.groundtruth_classes] = groundtruth_classes

        label_id_offset_paddings = tf.constant([[0, 0], [1, 0]])
        if fields.InputDataFields.groundtruth_verified_neg_classes in groundtruth:
            groundtruth[
                fields.InputDataFields.groundtruth_verified_neg_classes] = tf.pad(
                    groundtruth[
                        fields.InputDataFields.groundtruth_verified_neg_classes],
                    label_id_offset_paddings)
        if fields.InputDataFields.groundtruth_not_exhaustive_classes in groundtruth:
            groundtruth[
                fields.InputDataFields.groundtruth_not_exhaustive_classes] = tf.pad(
                    groundtruth[
                        fields.InputDataFields.groundtruth_not_exhaustive_classes],
                    label_id_offset_paddings)
        if fields.InputDataFields.groundtruth_labeled_classes in groundtruth:
            groundtruth[fields.InputDataFields.groundtruth_labeled_classes] = tf.pad(
                groundtruth[fields.InputDataFields.groundtruth_labeled_classes],
                label_id_offset_paddings)

        use_original_images = fields.InputDataFields.original_image in features
        if use_original_images:
            eval_images = features[fields.InputDataFields.original_image]
            true_image_shapes = features[fields.InputDataFields.true_image_shape][:, :3]
            original_image_spatial_shapes = features[
                fields.InputDataFields.original_image_spatial_shape]
        else:
            eval_images = features[fields.InputDataFields.image]
            true_image_shapes = None
            original_image_spatial_shapes = None
        eval_dict = result_dict_for_batched_example(
            eval_images,
            features[self.HASH_KEY],
            detections,
            groundtruth,
            class_agnostic=class_agnostic,
            scale_to_absolute=True,
            original_image_spatial_shapes=original_image_spatial_shapes,
            true_image_shapes=true_image_shapes)

        return eval_dict, class_agnostic

    def __call__(self, prediction_dict, features, labels):
        from object_detection.core import standard_fields as fields
        from object_detection.meta_architectures.center_net_meta_arch\
            import top_k_feature_map_locations, prediction_tensors_to_boxes,\
            convert_strided_predictions_to_normalized_boxes

        object_center_prob = tf.nn.sigmoid(
            prediction_dict[self.output_map["object_center"]])
        # Get x, y and channel indices corresponding to the top indices in the class
        # center predictions.
        detection_scores, y_indices, x_indices, channel_indices = (
            top_k_feature_map_locations(object_center_prob, max_pool_kernel_size=3,
                                        k=self.MAX_BOX_PREDICTIONS))
        multiclass_scores = tf.gather_nd(
            object_center_prob, tf.stack([y_indices, x_indices], -1), batch_dims=1)
        num_detections = tf.reduce_sum(
            tf.compat.v1.to_int32(detection_scores > 0), axis=1)
        local_prediction_dict = {
            fields.DetectionResultFields.detection_scores: detection_scores,
            fields.DetectionResultFields.detection_multiclass_scores: multiclass_scores,
            fields.DetectionResultFields.detection_classes: channel_indices,
            fields.DetectionResultFields.num_detections: num_detections}

        boxes_strided = (prediction_tensors_to_boxes(y_indices, x_indices,
                                                     prediction_dict[self.output_map["box_scale"]],
                                                     prediction_dict[self.output_map["box_offset"]]))
        true_image_shapes = features[fields.InputDataFields.true_image_shape][:, :3]
        boxes = convert_strided_predictions_to_normalized_boxes(boxes_strided,
                                                                self.STRIDE, true_image_shapes)
        local_prediction_dict.update({
            fields.DetectionResultFields.detection_boxes: boxes,
            'detection_boxes_strided': boxes_strided})

        def concat_replica_results(tensor_dict):
            new_tensor_dict = {}
            for key, values in tensor_dict.items():
                new_tensor_dict[key] = tf.concat(values, axis=0)
            return new_tensor_dict

        local_prediction_dict = concat_replica_results(local_prediction_dict)
        local_groundtruth_dict = labels
        local_groundtruth_dict = concat_replica_results(local_groundtruth_dict)
        local_eval_features = {
            fields.InputDataFields.image:
                features[fields.InputDataFields.image],
            fields.InputDataFields.original_image:
                features[fields.InputDataFields.original_image],
            fields.InputDataFields.original_image_spatial_shape:
                features[fields.InputDataFields.original_image_spatial_shape],
            fields.InputDataFields.true_image_shape:
                features[fields.InputDataFields.true_image_shape],
            self.HASH_KEY: features[self.HASH_KEY],
        }
        local_eval_features = concat_replica_results(local_eval_features)
        eval_dict, _ = self.prepare_eval_dict(local_prediction_dict,
                                              local_groundtruth_dict,
                                              local_eval_features)

        return eval_dict

def save_result(
    input_count,
    batch_size,
    map_acc,
    hardwaretime,
    endToEndTime,
    result_json,
    net_name,
    precision,
    mode,
):

    TIME = -1
    hardwareFps = -1
    hwLatencyTime = -1
    endToEndFps = -1
    e2eLatencyTime = -1
    if hardwaretime != TIME:
        hardwareFps = input_count / hardwaretime
        hwLatencyTime = hardwaretime / (input_count / batch_size) * 1000
    if endToEndTime != TIME:
        e2eLatencyTime = endToEndTime / (input_count / batch_size) * 1000
        endToEndFps = input_count / endToEndTime
    result = {
        "Output": {
            "Accuracy": {
                "map": "%.2f" % map_acc["map"],
            },
            "HostLatency(ms)": {
                "average": "%.2f" % e2eLatencyTime,
                "throughput(fps)": "%.2f" % endToEndFps,
            },
            "HardwareLatency(ms)": {
                "average": "%.2f" % hwLatencyTime,
                "throughput(fps)": "%.2f" % hardwareFps,
            },
            "BasicInfo": {
                "net_name": "%s" % net_name,
                "mode": "%s" % mode,
                "batch_size": "%d" % batch_size,
                "quant_precision": "%s" % precision,
            },
        }
    }

    tmp_res_1 = net_name +"\t"+ mode +"\t"
    tmp_res_2 = precision+"\t"+str(batch_size)+"\t"+ str(map_acc["map"]) + "\t" +str(hardwareFps)
    tmp_res = tmp_res_1 + tmp_res_2+"\n"
    if not os.path.exists(result_json):
        os.mknod(result_json)
    with open(result_json,"a") as f_obj:
        f_obj.write(tmp_res)
