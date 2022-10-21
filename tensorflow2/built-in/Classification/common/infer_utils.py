import os

def decode_predictions(preds, top=5):
    """Decodes the prediction of an ImageNet model.
    Args:
      preds: Numpy array encoding a batch of predictions.
      top: Integer, how many top-guesses to return. Defaults to 5.
    Returns:
      A list of lists of top class prediction tuples
      `(class_name, class_description, score)`.
      One list of tuples per sample in batch input.
    Raises:
      ValueError: In case of invalid shape of the `pred` array
        (must be 2D).
    """

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError(
            "`decode_predictions` expects "
            "a batch of predictions "
            "(i.e. a 2D array of shape (samples, 1000)). "
            "Found array with shape: " + str(preds.shape)
        )
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(str(i), pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def calculate_topk_ratio(top1_sum, top5_sum, img_cnt):
    if img_cnt > 0:
        top1_ratio = top1_sum / img_cnt * 1.0
        top5_ratio = top5_sum / img_cnt * 1.0
        return (top1_ratio, top5_ratio)
    else:
        print("Error!denominator is 0!")
        exit(1)


def calculate_accuracy(y_pred, y_true):
    # traverse the data_dict and the decode_res, to check
    # whether the pred_res == label
    # Notice: the data_dict may contain multi-batch data
    top1_cnt = 0
    top5_cnt = 0
    total_cnt = 0
    for img_label, decode_res in zip(y_true, y_pred):
        decode_label_list = []
        img_label = str(img_label.numpy()[0])
        total_cnt += 1
        for idx, item in enumerate(decode_res):
            decode_label_list.append(str(item[0]))
        if img_label == decode_label_list[0]:
            top1_cnt += 1
            top5_cnt += 1
        elif img_label in decode_label_list:
            top5_cnt += 1
    return (top1_cnt, top5_cnt, total_cnt)


def accuracy(output, target, last_batch=0, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0) if last_batch == 0 else last_batch

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    if last_batch > 0:
        _, pred = output[0:last_batch].topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[0:last_batch].view(1, -1).expand_as(pred))

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def save_result(
    imageNum,
    batch_size,
    top1,
    top5,
    meanAp,
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
        hardwareFps = imageNum / hardwaretime
        hwLatencyTime = hardwaretime / (imageNum / batch_size) * 1000
    if endToEndTime != TIME:
        e2eLatencyTime = endToEndTime / (imageNum / batch_size) * 1000
        endToEndFps = imageNum / endToEndTime
    result = {
        "Output": {
            "Accuracy": {
                "top1": "%.2f" % top1,
                "top5": "%.2f" % top5,
                "meanAp": "%.2f" % meanAp,
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
    tmp_res_2 = precision+"\t"+str(batch_size)+"\t"+ str(top1) + "/" + str(top5) + "\t" +str(hardwareFps)
    tmp_res = tmp_res_1 + tmp_res_2+"\n"
    if not os.path.exists(result_json):
        os.mknod(result_json)
    with open(result_json,"a") as f_obj:
        f_obj.write(tmp_res)


def generate_img_label_map(img_label_file):
    img_label_map = {}
    with open(img_label_file, "r") as f_map:
        for line in f_map.readlines():
            line = line.strip("\n").split(" ")
            img_label_map[line[0]] = line[1]
    return img_label_map


def get_img_label(img_path, img_label_map):
    img_name = str(img_path).split("/")[-1]
    if img_name in img_label_map:
        return img_label_map[img_name]
    else:
        return "None"
