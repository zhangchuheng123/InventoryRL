import os
import pdb
import pandas as pd
from scipy import interpolate
from tensorboard.backend.event_processing import event_accumulator

def main(log_dir='logs/', output_dir='results/'):

	os.makedirs(output_dir, exist_ok=True)

	process_ss(log_dir, output_dir)
	process_sacd(log_dir, output_dir)

def process_ss(log_dir, output_dir, budget=30):

	paths = os.listdir(log_dir)

	record = []
	for ppl in [2, 4, 6]:
		for algo in ['ss_policy']:

			print('Processing ppl (P+L): {} algo: {}'.format(ppl, algo))

			target_path = [item for item in paths if 'ppl{}'.format(ppl) in item and algo in item][0]
			target_file = os.path.join(log_dir, target_path, 'record', 'stats.csv')

			df = pd.read_csv(target_file, index_col=0)
			f = interpolate.interp1d(df['budget'].values, df['cost'].values)
			record.append(dict(ppl=ppl, budget=budget, cost=f(budget)))

	pd.DataFrame(record).to_csv(os.path.join(output_dir, 'ss_policy.csv'))

def process_sacd(log_dir, output_dir):

	paths = os.listdir(log_dir)

	for ppl in [2, 4, 6]:
		for algo in ['sacd_ours', 'sacd_bench']:

			print('Processing ppl (P+L): {} algo: {}'.format(ppl, algo))

			target_path = [item for item in paths if 'ppl{}'.format(ppl) in item and algo in item][0]
			target_path = os.path.join(log_dir, target_path, 'summary')
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

			target_output_dir = os.path.join(output_dir, 'ppl{}_{}'.format(ppl, algo))
			os.makedirs(target_output_dir, exist_ok=True)

			pd.DataFrame(ea.Scalars('eval/return_discount')).to_csv(os.path.join(target_output_dir, 'return_discount.csv'))
			pd.DataFrame(ea.Scalars('eval/budget_discount')).to_csv(os.path.join(target_output_dir, 'budget_discount.csv'))

if __name__ == '__main__':

	main()