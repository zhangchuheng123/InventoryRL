import os
import pdb
import pandas as pd
from scipy import interpolate
from tensorboard.backend.event_processing import event_accumulator

def main(dir_log='logs/', output_path='results/'):

	os.makedirs(output_path, exist_ok=True)

	process_ss(log_dir, output_path)
	process_sacd(log_dir, output_path)

def process_ss(log_dir, output_path, budget=30):

	paths = os.listdir(dir_log)

	record = []
	for ppl in [2, 4, 6]:
		for algo in ['ss_policy']:

			target_path = [item for item in paths if 'ppl{}'.format(ppl) in item and algo in item][0]
			target_file = os.path.join(dir_log, target_path, 'record', 'stats.csv')

			df = pd.read_csv(target_file, index_col=0)
			f = interpolate.interp1d(df['budget'].values, df['cost'].values)
			record.append(dict(ppl=ppl, budget=budget, cost=f(budget)))

	pd.DataFrame(record).to_csv(os.path.join(target_output_path, 'ss_policy.csv'))

def process_sacd(log_dir, output_path):

	paths = os.listdir(dir_log)

	for ppl in [2, 4, 6]:
		for algo in ['sacd_ours', 'sacd_bench']:

			target_path = [item for item in paths if 'ppl{}'.format(ppl) in item and algo in item][0]
			target_path = os.path.join(dir_log, target_path, 'summary')
			target_file = [item for item in os.listdir(target_path) if 'tfevents' in item][0]
			target_file = os.path.join(target_path, target_file)

			ea = event_accumulator.EventAccumulator(target_file, size_guidance={
				event_accumulator.COMPRESSED_HISTOGRAMS: 500,
				event_accumulator.IMAGES: 4,
				event_accumulator.AUDIO: 4,
				event_accumulator.SCALARS: 0,
				event_accumulator.HISTOGRAMS: 1})
			# This could be very slow
			ea.Reload()

			pdb.set_trace()

			target_output_path = os.path.join(output_path, 'ppl{}_{}'.format(ppl, algo))
			os.makedirs(target_output_path, exist_ok=True)

			pd.DataFrame(ea.Scalars('eval/return_discount')).to_csv(os.path.join(target_output_path, 'return_discount.csv'))
			pd.DataFrame(ea.Scalars('eval/budget_discount')).to_csv(os.path.join(target_output_path, 'budget_discount.csv'))

if __name__ == '__main__':

	main()