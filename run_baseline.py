import torch, random, os, pdb, copy, random, pickle, agents
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.nn import functional as F
from torch.autograd import grad, Variable, Function
from agym.envs import SpaceShip
from collections import deque
from itertools import chain
from tqdm import tqdm


if __name__ == '__main__':
	MAX_RUN = 128
	MAX_EPOCH = 60000
	SAMPLE_SIZE = 1024
	LR = 1e-3
	SAVE_INTERVAL = 1000
	OUTLET_BASE = 'outlet/baseline/run-{:03d}'
	# load training and testing sets
	trnset = np.load('/home/yen/data/frontier-boredom/single-step/trn_dataset_nonseq.npy').tolist()
	tesset = np.load('/home/yen/data/frontier-boredom/single-step/tes_dataset_nonseq.npy').tolist()


	for run in range(MAX_RUN):
		OUTLET_PATH = OUTLET_BASE.format(run + 1)
		if os.path.isdir(OUTLET_PATH):
			print('Skipping run {}'.format(run + 1))
			continue
		else:
			os.mkdir(OUTLET_PATH)
			os.mkdir(os.path.join(OUTLET_PATH, 'models'))

		# starts a new run with new agent
		agnt = agents.AgentBASELINE().cuda()
		adam = optim.Adam(agnt.parameters(), LR)
		schd = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='min', factor=0.5, patience=10, verbose=True)

		total_trnloss = 0
		total_tesloss = 0
		with open(os.path.join(OUTLET_PATH, 'trnloss.txt'), 'wb') as fd_trnloss, \
			 open(os.path.join(OUTLET_PATH, 'tesloss.txt'), 'wb') as fd_tesloss, \
			 tqdm(range(MAX_EPOCH)) as pbar:
			pbar.set_description('=Run-{}='.format(run + 1))
			for epoch in pbar:
				trn_s0, trn_a0, trn_s1 = map(lambda o: Variable(torch.FloatTensor(o)).cuda(), np.split(random.sample(trnset, SAMPLE_SIZE), [4, 6, 10], axis=1)[:-1])
				tes_s0, tes_a0, tes_s1 = map(lambda o: Variable(torch.FloatTensor(o), volatile=True).cuda(), np.split(random.sample(tesset, SAMPLE_SIZE), [4, 6, 10], axis=1)[:-1])
				trnloss = (agnt.eval_fm(trn_s0, trn_a0)[0] - trn_s1).pow(2).sum(1).mean()
				tesloss = (agnt.eval_fm(trn_s0, trn_a0)[0] - trn_s1).pow(2).sum(1).mean()
				total_trnloss += float(trnloss)
				total_tesloss += float(tesloss)
				trnloss.backward()
				adam.step()
				adam.zero_grad()
				if (epoch + 1) == 1 or (epoch + 1) % SAVE_INTERVAL == 0:
					torch.save(agnt.state_dict(), os.path.join(OUTLET_PATH, 'models/blm_{:07d}.pt'.format(epoch + 1)))

				if (epoch + 1) % 60 == 0:
					schd.step(total_tesloss)
					np.savetxt(fd_trnloss, [total_trnloss / 60]), fd_trnloss.flush()
					np.savetxt(fd_tesloss, [total_tesloss / 60]), fd_tesloss.flush()
					total_trnloss, total_tesloss = 0, 0
