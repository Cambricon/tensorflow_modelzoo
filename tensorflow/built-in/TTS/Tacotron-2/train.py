import argparse
import os
from time import sleep

import tensorflow as tf
import infolog
from infolog import log
from hparams import hparams, hparams_debug_string
from wavenet_vocoder.synthesize import tacotron_synthesize
from wavenet_vocoder.train import tacotron_train
from wavenet_vocoder.train import wavenet_train

def save_seq(file, sequence, input_path):
	'''Save Tacotron-2 training state to disk. (To skip for future runs)
	'''
	sequence = [str(int(s)) for s in sequence] + [input_path]
	with open(file, 'w') as f:
		f.write('|'.join(sequence))

def read_seq(file):
	'''Load Tacotron-2 training state from disk. (To skip if not first run)
	'''
	if os.path.isfile(file):
		with open(file, 'r') as f:
			sequence = f.read().split('|')

		return [bool(int(s)) for s in sequence[:-1]], sequence[-1]
	else:
		return [0, 0, 0], ''

def prepare_run(args):
	hp = "tacotron_num_devices={}, tacotron_batch_size={}, tacotron_synthesis_batch_size={}, device_type={}, use_amp={}, use_horovod={}, use_profiler={}, use_performance={}".format(args.tacotron_num_devices, args.tacotron_batch_size, args.tacotron_num_devices, args.device_type, args.use_amp, args.use_horovod, args.use_profiler, args.use_performance)
	#modified_hp = hparams.parse(args.hparams)
	modified_hp = hparams.parse(hp)
	run_name = args.name or args.model
	log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
	os.makedirs(log_dir, exist_ok=True)
	infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name, args.slack_url)
	return log_dir, modified_hp

def train(args, log_dir, hparams):
	state_file = os.path.join(log_dir, 'state_log')
	#Get training states
	(taco_state, GTA_state, wave_state), input_path = read_seq(state_file)

	if not taco_state:
		log('\n#############################################################\n')
		log('Tacotron Train\n')
		log('###########################################################\n')
		checkpoint = tacotron_train(args, log_dir, hparams)
		tf.reset_default_graph()
		#Sleep 1/2 second to let previous graph close and avoid error messages while synthesis
		sleep(0.5)
		if checkpoint is None:
			raise ValueError('Error occured while training Tacotron, Exiting!')
		taco_state = 1
		save_seq(state_file, [taco_state, GTA_state, wave_state], input_path)
	else:
		checkpoint = os.path.join(log_dir, 'taco_pretrained/')

	if taco_state:
		log('TRAINING IS ALREADY COMPLETE!!')

def main():
	def str2bool(v):
            parser = argparse.ArgumentParser()
            if isinstance(v, bool):
                return v
            elif v.lower() in ('true', '1'):
                return True
            elif v.lower() in ('false', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--tacotron_num_devices', type=int, default=1, help='Determines the number of devices in use for Tacotron training.')
	parser.add_argument('--tacotron_batch_size', type=int, default=32, help='number of training samples on each training steps')
	parser.add_argument('--tacotron_synthesis_batch_size', type=int, default=1, help='number of batchsize in eval')
	parser.add_argument('--device_type', default='mlu', help='which device to use, must be mlu/gpu/cpu.')
	parser.add_argument('--tacotron_input', default='training_data/train.txt')
	parser.add_argument('--wavenet_input', default='tacotron_output/gta/map.txt')
	parser.add_argument('--name', help='Name of logging directory.')
	parser.add_argument('--model', default='Tacotron-2')
	parser.add_argument('--input_dir', default='training_data', help='folder to contain inputs sentences/targets')
	parser.add_argument('--taco_checkpoint', default='', help='Path to taco model checkpoint')
	parser.add_argument('--output_dir', default='output', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--mode', default='synthesis', help='mode for synthesis of tacotron after training')
	parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode')
	parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
	parser.add_argument('--summary_interval', type=int, default=250,
		help='Steps between running summary ops')
	parser.add_argument('--embedding_interval', type=int, default=10000,
		help='Steps between updating embeddings projection visualization')
	parser.add_argument('--checkpoint_interval', type=int, default=5000,
		help='Steps between writing checkpoints')
	parser.add_argument('--eval_interval', type=int, default=10000,
		help='Steps between eval on test data')
	parser.add_argument('--tacotron_train_steps', type=int, default=150000, help='total number of tacotron training steps')
	parser.add_argument('--wavenet_train_steps', type=int, default=750000, help='total number of wavenet training steps')
	parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
	parser.add_argument('--slack_url', default=None, help='slack webhook notification destination link')
	parser.add_argument('--use_amp', type=str2bool, default=False, help='If use amp, please set True.')
	parser.add_argument('--use_horovod', type=str2bool, default=False, help='If use horovod, please set True.')
	parser.add_argument('--use_profiler', type=str2bool, default=False, help='Use profiler to train nets or not.')
	parser.add_argument('--use_performance', type=str2bool, default=False, help='Use performance tools to get fps or not.')
	args = parser.parse_args()

	accepted_models = ['Tacotron', 'WaveNet', 'Tacotron-2']

	if args.model not in accepted_models:
		raise ValueError('please enter a valid model to train: {}'.format(accepted_models))

	if args.use_horovod:
		import horovod.tensorflow as hvd
		global hvd
		hvd.init()

	log_dir, hparams = prepare_run(args)
	if args.model == 'Tacotron':
		tacotron_train(args, log_dir, hparams)
	elif args.model == 'WaveNet':
		wavenet_train(args, log_dir, hparams, args.wavenet_input)
	elif args.model == 'Tacotron-2':
		train(args, log_dir, hparams)
	else:
		raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))


if __name__ == '__main__':
	main()
