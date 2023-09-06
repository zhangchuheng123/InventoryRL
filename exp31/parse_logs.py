import os
import pdb
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def main(dir_log='logs/'):

	paths = os.listdir(dir_log)
	for ppl in [2, 4, 6]:
		for algo in ['sacd_ours', 'sacd_bench', 'ss_policy']:

			target_path = [item for item in paths if 'ppl{}'.format(ppl) in item and algo in item][0]

			if algo == 'ss_policy':

				pass

			else:

				target_path = os.path.join(dir_log, target_path, 'summary')
				target_file = os.listdir(target_path)
				target_file = os.path.join(target_path, target_file)

				ea = event_accumulator.EventAccumulator(target_file, size_guidance={
					event_accumulator.COMPRESSED_HISTOGRAMS: 500,
					event_accumulator.IMAGES: 4,
					event_accumulator.AUDIO: 4,
					event_accumulator.SCALARS: 0,
					event_accumulator.HISTOGRAMS: 1})

				pdb.set_trace()

				df = pd.DataFrame(ea.Scalars('eval'))
