import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB

import logging
from pytorch_worker import PyTorchWorker as worker

logging.basicConfig(level=logging.DEBUG)

min_budget = 3
max_budget = 30
n_iterations = 4
# worker = True # ??
nic_name = 'lo'
run_id = 'ResNet_featExtr'
shared_directory = './drive/MyDrive/HW2/ResNet_FE/'

# Every process has to lookup the hostname
host = hpns.nic_name_to_host(nic_name)
print(host)


print('here')
# This example shows how to log live results. This is most useful
# for really long runs, where intermediate results could already be
# interesting. The core.result submodule contains the functionality to
# read the two generated files (results.json and configs.json) and
# create a Result object.
result_logger = hpres.json_result_logger(directory=shared_directory, overwrite=False)


# Start a nameserver:
NS = hpns.NameServer(run_id=run_id, host=host, port=0, working_directory=shared_directory)
ns_host, ns_port = NS.start()

# Start local worker
w = worker(run_id=run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120)
w.run(background=True)

# Run an optimizer
bohb = BOHB(  configspace = worker.get_configspace(),
			  run_id = run_id,
			  host=host,
			  nameserver=ns_host,
			  nameserver_port=ns_port,
			  result_logger=result_logger,
			  min_budget=min_budget, max_budget=max_budget,
		   )
res = bohb.run(n_iterations=n_iterations)

# store results
with open(os.path.join(shared_directory, 'results.pkl'), 'wb') as fh:
	pickle.dump(res, fh)

# shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()
