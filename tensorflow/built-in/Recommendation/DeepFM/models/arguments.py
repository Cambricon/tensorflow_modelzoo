import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

PARSER = argparse.ArgumentParser(description="DeepFM")

# Training parameters flags
PARSER.add_argument('--batch_size', default=1024, type=int, help='number of training samples on each training steps')
PARSER.add_argument('--exec_mode', choices=['DeepFM', 'DNN', 'FM'], type=str)
PARSER.add_argument('--epoch', default=30, type=int)
PARSER.add_argument('--embedding_size', default=8, type=int)
PARSER.add_argument('--optimizer_type', choices=['adam', 'adagrad', 'gd', 'momentum'], default='adam', type=str)
PARSER.add_argument('--learning_rate', default=0.001, type=float)
PARSER.add_argument('--model_dir', required=True, type=str)
PARSER.add_argument('--ckpt_file', default='', type=str)
PARSER.add_argument('--data_dir', required=True, type=str)
PARSER.add_argument('--finetune_steps', default=0, type=int)
PARSER.add_argument('--num_splits', default=3, type=int)

# Training configuration flags
PARSER.add_argument('--use_gpu', default=False, type=str2bool)
PARSER.add_argument('--use_amp', default=False, type=str2bool)
PARSER.add_argument('--use_performance', default=False, type=str2bool)
PARSER.add_argument('--use_profiler', default=False, type=str2bool)
PARSER.add_argument('--use_horovod', default=False, type=str2bool)
PARSER.add_argument('--verbose', default=True, type=str2bool)
PARSER.add_argument('--skip_eval', default=False, type=str2bool)
