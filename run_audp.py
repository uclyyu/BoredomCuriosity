# audp, agent with uniformly distributed policy
# replace pgur policy network with uniform samples from [0, 255)
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
from glob import glob


def _isfreestate(s0, s1):
	if (s0 == s1).prod() == 1:
		return [0]
	return [1]

def step_fill(env, agnt, buf, asp, cuda=False):
	# sample action, step, and fill DB
	s0 = env.state
	i0 = int(np.random.uniform() * 121)
	a0 = asp[i0]
	s1 = env.step(a0)
	isfree = _isfreestate(s0, s1)	
	buf.append(np.concatenate([s0, a0, [i0], s1, isfree], axis=0))
	return isfree[0]

def db2list(db):
	return list(chain(*db))

def _sampleDB(dblist, n, cuda=False):
	samp = np.array(random.sample(dblist, n))
	s0, a0, i0, s1, _ = map(lambda o: Variable(torch.from_numpy(o).float()), np.split(samp, [4, 6, 7, 11], axis=1))
	if cuda:
		s0, a0, i0, s1 = map(lambda o: o.cuda(), [s0, a0, i0, s1])
	return s0, a0, i0, s1

def fit_fm(roll, agnt, ctl, dblist, n, cuda=False):
	s0, a0, i0, s1 = _sampleDB(dblist, n, cuda=cuda)
	loss = (agnt.eval_fm(s0, a0)[0] - s1).pow(2).sum(1).mean()
	ctl.register_loss(loss)
	loss.backward()
		
	ctl.step()
	if (roll + 1) % 300 == 0:
		ctl.lr_step()

def main(run, ob, asp):
	LR = 5e-5  # world-model larning rate
	DB_EPS = 100  # number of episode to hold in experience
	INIT_EPS = 99
	N_PREROLL = 30000
	N_POSTROLL = 30000
	MAX_EPILEN = 300  # episode length
	SAMPLE_SIZE = 64
	SAVE_INTERVAL = 1000
	OUTLET_BASE = ob
	experienceDB = deque([], maxlen=DB_EPS)
	episodic_buf = []

	OUTLET_PATH = OUTLET_BASE.format(run + 1)
	if os.path.isdir(OUTLET_PATH):
		print('Skipping run {}'.format(run + 1))
		return
	else:
		os.mkdir(OUTLET_PATH)
		os.mkdir(os.path.join(OUTLET_PATH, 'models'))
		os.mkdir(os.path.join(OUTLET_PATH, 'experiences'))

	# starts a new run with new agent
	agnt = agents.AgentAUDP().cuda()

	# fill initial experience 
	for epi in tqdm(range(INIT_EPS)):
		env = SpaceShip()
		episodic_buf = []
		experienceDB.append(episodic_buf)
		while step_fill(env, agnt, episodic_buf, asp, cuda=True):
			if len(episodic_buf) >= MAX_EPILEN:
				break
	with open(os.path.join(OUTLET_PATH, 'init_experienceDB.pkl'), 'wb') as pkl:
		pickle.dump(experienceDB, pkl)

	with open(os.path.join(OUTLET_PATH, 'fm_loss.txt'), 'wb') as fd_fm_loss, \
		 open(os.path.join(OUTLET_PATH, 'episode_len.txt'), 'wb') as fd_epilen, \
		 open(os.path.join(OUTLET_PATH, 'episode_aen.txt'), 'wb') as fd_epiaen, \
		 tqdm(range(N_PREROLL)) as pre_pbar, \
		 tqdm(range(N_PREROLL, N_POSTROLL + N_PREROLL)) as post_pbar:
		
			ctl_fm = agents.UpdateControl(fd_fm_loss, agnt.fm.parameters(), LR, ignore=True)  # ignore scheduler
			dbl = db2list(experienceDB)
			env = SpaceShip()
			episodic_buf = []
			episode = 0
			experienceDB.append(episodic_buf)
			pre_pbar.set_description('=Run-{}='.format(run + 1))
			
			for rollout in pre_pbar:
				# step and fill episodic buffer
				if step_fill(env, agnt, episodic_buf, action_space, cuda=True):
					pre_pbar.set_description('=Run-{}=PreRoll-{}.Go='.format(run + 1, rollout + 1))
				else:
					pre_pbar.set_description('=Run-{}=PreRoll-{}.Nogo='.format(run + 1, rollout + 1))
						
				# save model on rollout
				if (rollout + 1) == 1 or (rollout + 1) % SAVE_INTERVAL == 0:
					torch.save(agnt.state_dict(), os.path.join(OUTLET_PATH, 'models/agnt_{:07d}.pt'.format(rollout + 1)))
				
				# fit model on rollout
				dbl = db2list(experienceDB)
				fit_fm(rollout, agnt, ctl_fm, dbl, SAMPLE_SIZE, cuda=True)
				
				# collect episodic information
				if (rollout + 1) % MAX_EPILEN == 0:
					episode += 1
					# save episodic experience
					np.savetxt(fd_epilen, [len(episodic_buf)]), fd_epilen.flush()
					np.save(os.path.join(OUTLET_PATH, 'experiences/exp_{:07d}.npy'.format(episode)), episodic_buf)

					# save action entropy information
					# [0] entropy of distribution over emitted actions in one episode
					expn = np.load(os.path.join(OUTLET_PATH, 'experiences/exp_{:07d}.npy'.format(episode)))
					dfrm = pd.DataFrame({'ai': expn[:, 6].astype('int'), 'c': 1})
					cntr = dfrm.groupby(['ai']).count()
					dist = np.zeros(121)
					dist[cntr.index] = np.array(cntr).ravel()
					dist = dist / dist.sum()
					with np.warnings.catch_warnings():
						np.warnings.filterwarnings('ignore')
						entt = np.where(dist > 0, -np.log(dist) * dist, 0).sum()
					np.savetxt(fd_epiaen, [[entt]]), fd_epiaen.flush()
					
					if (rollout + 1) < N_PREROLL:
						# new episodic buffer
						episodic_buf = []
						experienceDB.append(episodic_buf)

			# Train world model after accumulating experience
			dbl = db2list(experienceDB)
			post_pbar.set_description('=Run-{}=PostData Phase='.format(run + 1))
			ctl_fm.ignore_scheduler(False)
			for rollout in post_pbar:
				if (rollout + 1) % SAVE_INTERVAL == 0:
					torch.save(agnt.state_dict(), os.path.join(OUTLET_PATH, 'models/agnt_{:07d}.pt'.format(rollout + 1)))
				fit_fm(rollout, agnt, ctl_fm, dbl, SAMPLE_SIZE, cuda=True)

if __name__ == '__main__':
	MAX_RUN = 128
	OUTLET_BASE = 'outlet/audp/run-{:03d}'
	action_space = np.mgrid[-2:2:41j, -2:2:41j].reshape([2, -1]).T

	for run in range(MAX_RUN):
		try:
			main(run, OUTLET_BASE, action_space)
		except Exception as exc:
			import logging, traceback
			print('\nRun-{} failed.'.format(run + 1))
			logging.error(traceback.format_exc())
			print('-' * 80)
