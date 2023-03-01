import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

from models import  config
from models.metrics import gini_norm
from models.DataReader import FeatureDictionary, DataParser
from models.DeepFM import DeepFM
from models.arguments import PARSER

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)
params = PARSER.parse_args()


def _load_data():

    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values
    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices


def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        # The multiplication is so that the data set is not empty when testing the performance pipeline
        if params.use_horovod and (params.use_performance or params.use_profiler):
            num = hvd.size()
            Xi_train_ = Xi_train_ * num
            Xv_train_= Xv_train_ * num
            y_train_ = y_train_ * num
        if params.use_horovod:
            num_examples_per_rank = len(Xi_train_) // hvd.size()
            remainder = len(Xi_train_) % hvd.size()
            if hvd.rank() < remainder:
                start_index = hvd.rank() * (num_examples_per_rank+1)
                end_index = start_index + num_examples_per_rank + 1
            else:
                start_index = hvd.rank() * num_examples_per_rank + remainder
                end_index = start_index + (num_examples_per_rank)
            Xi_train_, Xv_train_, y_train_ = Xi_train_[start_index:end_index], Xv_train_[start_index:end_index], y_train_[start_index:end_index]
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        if params.skip_eval:
            dfm.fit(Xi_train=Xi_train_, Xv_train=Xv_train_, y_train=y_train_)
        else:
            dfm.fit(Xi_train=Xi_train_, Xv_train=Xv_train_, y_train=y_train_, Xi_valid=Xi_valid_, Xv_valid=Xv_valid_,
                    y_valid=y_valid_)

            y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_, Xv_valid_)
            y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test)

            gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
            gini_results_epoch_train[i] = dfm.train_result
            if not params.use_horovod or (params.use_horovod and hvd.rank() == 0):
                gini_results_epoch_valid[i] = dfm.valid_result
        if params.use_performance or params.use_profiler:
            break
    print("gini_results_cv：", gini_results_cv)

    y_test_meta /= float(len(folds))

    # save result
    if (params.use_horovod and hvd.rank() == 0) or (not params.use_horovod):
        if dfm_params["use_fm"] and dfm_params["use_deep"]:
            clf_str = "DeepFM"
        elif dfm_params["use_fm"]:
            clf_str = "FM"
        elif dfm_params["use_deep"]:
            clf_str = "DNN"
        print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
        filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())
        _make_submission(ids_test, y_test_meta, filename)

        _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    if (params.use_horovod and hvd.rank() == 0) or (not params.use_horovod):
        pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
            os.path.join(params.model_dir, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    if not os.path.exists("./fig"):
        os.mkdir("./fig")
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png"%model_name)
    plt.close()


def main():

    if params.use_horovod:
        import horovod.tensorflow as hvd
        global hvd
        hvd.init()

    if params.use_performance and params.use_profiler:
        raise ValueError("You can only set use_profiler or use_performance, not at the same time, otherwise the e2e time will be worse")

    if not os.path.exists(params.model_dir) and (not params.use_horovod or (params.use_horovod and hvd.rank() == 0)):
        os.mkdir(params.model_dir)

    # load data
    dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()

    # folds
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))

    # params
    network_params = {
        "use_fm": True,
        "use_deep": True,
        "embedding_size": params.embedding_size,
        "dropout_fm": [1.0, 1.0],
        "deep_layers": [32, 32],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": params.epoch,
        "batch_size": params.batch_size,
        "learning_rate": params.learning_rate,
        "optimizer_type": params.optimizer_type,
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": params.verbose,
        "eval_metric": gini_norm,
        "random_seed": config.RANDOM_SEED
    }

    if "DeepFM" == params.exec_mode:
        y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, network_params)

    if "FM" == params.exec_mode:
        network_params["use_deep"] = False
        y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, network_params)

    if "DNN" == params.exec_mode:
        network_params["use_fm"] = False
        y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, network_params)


if __name__ == '__main__':
	main()
