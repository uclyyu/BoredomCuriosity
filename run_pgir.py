# pgir, policy gradients from intrinsic reward sample
# replace pedl reward and value functions altogether with Unif(-1, 1) samples.
import torch, random, os, pdb, copy, random, pickle, agents
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import pandas as pd
from torch.nn import functional as F
from torch.autograd import grad, Variable, Function
from agym.envs import SpaceShip
from collections import deque
from itertools import chain, product
from tqdm import tqdm
from glob import glob
from scipy.stats import norm

norm_dist = norm(loc=0.00060765360735822469, scale=0.01346543415649919)

def _isfreestate(s0, s1):
	# sliding aginst boundary is considered free
	if (s0 == s1).prod() == 1 and (s1[0] == 0 or s1[0] == 1) and (s1[1] == 0 or s1[1] == 1):
		return [0]
	return [1]
	
def step_fill(env, agnt, buf, cuda=False):
	# sample action, step, and fill DB
	s0 = env.state
	s0_pt = Variable(torch.from_numpy(s0).float().unsqueeze(0))
	if cuda:
		s0_pt = s0_pt.cuda()
	a0_pt, i0_pt, p0_pt = agnt.query_action(s0_pt)
	a0, i0, p0 = map(lambda o: o.data.cpu().numpy().ravel(), [a0_pt, i0_pt, p0_pt])
	s1 = env.step(a0)
	isfree = _isfreestate(s0, s1)
	buf.append(np.concatenate([s0, a0, i0, p0, s1, isfree], axis=0))
	return isfree[0]
	
def db2list(db):
	return list(chain(*db))
	
def _sampleDB(dblist, n, cuda=False):
	samp = np.array(random.sample(dblist, n))
	s0, a0, i0, p0, s1, gate = map(lambda o: Variable(torch.from_numpy(o).float()), np.split(samp, [4, 6, 7, 8, 12], axis=1))
	i0 = i0.long()
	if cuda:
		s0, a0, i0, p0, s1, gate = map(lambda o: o.cuda(), [s0, a0, i0, p0, s1, gate])
	return s0, a0, i0, p0, s1, gate

def _sampleIR(irlist, n, cuda=False):
	# samp = np.array(random.sample(irlist, n))
	samp = np.array(random.sample(irlist, n))
	ir = Variable(torch.FloatTensor(samp).view(-1, 1))
	if cuda:
		ir = ir.cuda()
	return ir

def fit_fm(roll, agnt, ctl, dblist, n, cuda=False):
	s0, a0, i0, p0, s1, gate = _sampleDB(dblist, n, cuda=cuda)
	loss = (agnt.eval_fm(s0, a0)[0] - s1).pow(2).sum(1).mean()
	ctl.register_loss(loss)
	for p, g in zip(agnt.fm.parameters(), grad(loss, agnt.fm.parameters())):
		if p.grad is None:
			p.grad = g
		else:
			p.grad += g
	ctl.step()
	if (roll + 1) % 300 == 0:
		ctl.lr_step()

def fit_ap(roll, agnt, ctl, dblist, irlist, n, cuda=False):
	s0, a0, i0, p0, s1, gate = _sampleDB(dblist, n, cuda=cuda)
	r1 = _sampleIR(irlist, n, cuda=cuda)
	q0 = agnt.infer_action_prob(s0, i0).add(1e-7)
	# ratio = agnt.infer_action_prob(s0, i0) / p0
	# clip_ratio = torch.clamp(ratio, 0.8, 1.2)
	# loss = -torch.min(ratio * r1, clip_ratio * r1).sum()
	loss = (-q0.log() * r1).sum()
	ctl.register_loss(loss)
	for p, g in zip(agnt.ap.parameters(), grad(loss, agnt.ap.parameters())):
		if p.grad is None:
			p.grad = g
		else:
			p.grad += g
	ctl.step()
	if (roll + 1) % 300 == 0:
		ctl.lr_step()

def main(run, ob, irs):
	LR_FM = 5e-5  # world-model larning rate
	LR_AP = 5e-5  # action policy learning rate
	DB_EPS = 100  # number of episode to hold in experience
	INIT_EPS = 99
	N_PREROLL = 30000
	N_POSTROLL = 30000
	MAX_EPILEN = 300  # episode length
	SAMPLE_SIZE = 64
	SAVE_INTERVAL = 1000
	IRSAMPLE = irs
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
	agnt = agents.AgentPGUR().cuda()

	# fill initial experience 
	for epi in tqdm(range(INIT_EPS)):
		env = SpaceShip()
		episodic_buf = []
		experienceDB.append(episodic_buf)
		while step_fill(env, agnt, episodic_buf, cuda=True):
			if len(episodic_buf) >= MAX_EPILEN:
				break
	with open(os.path.join(OUTLET_PATH, 'init_experienceDB.pkl'), 'wb') as pkl:
		pickle.dump(experienceDB, pkl)

	with open(os.path.join(OUTLET_PATH, 'fm_loss.txt'), 'wb') as fd_fm_loss, \
		 open(os.path.join(OUTLET_PATH, 'ap_loss.txt'), 'wb') as fd_ap_loss, \
		 open(os.path.join(OUTLET_PATH, 'episode_aen.txt'), 'wb') as fd_epiaen, \
		 open(os.path.join(OUTLET_PATH, 'episode_rav.txt'), 'wb') as fd_epirav, \
		 open(os.path.join(OUTLET_PATH, 'episode_len.txt'), 'wb') as fd_epilen, \
		 tqdm(range(N_PREROLL)) as pre_pbar, \
		 tqdm(range(N_PREROLL, N_POSTROLL + N_PREROLL)) as post_pbar:
		
			ctl_fm = agents.UpdateControl(fd_fm_loss, agnt.fm.parameters(), LR_FM, ignore=True)
			ctl_ap = agents.UpdateControl(fd_ap_loss, agnt.ap.parameters(), LR_AP, ignore=True)
			dbl = db2list(experienceDB)
			env = SpaceShip()
			episodic_buf = []
			episode = 0
			experienceDB.append(episodic_buf)
			
			for rollout in pre_pbar:
				# step and fill episodic memory
				if step_fill(env, agnt, episodic_buf, cuda=True):
					pre_pbar.set_description('=Run-{}=PreRoll-{}.Go='.format(run + 1, rollout + 1))
				else:
					pre_pbar.set_description('=Run-{}=PreRoll-{}.Nogo='.format(run + 1, rollout + 1))
						
				# save model on rollout
				if (rollout + 1) == 1 or (rollout + 1) % SAVE_INTERVAL == 0:
					torch.save(agnt.state_dict(), os.path.join(OUTLET_PATH, 'models/agnt_{:07d}.pt'.format(rollout + 1)))
				
				# fit model on rollout
				dbl = db2list(experienceDB)
				irl = IRSAMPLE[rollout]
				fit_fm(rollout, agnt, ctl_fm, dbl,      SAMPLE_SIZE, cuda=True)
				fit_ap(rollout, agnt, ctl_ap, dbl, irl, SAMPLE_SIZE, cuda=True)
				
				# collect episodic information
				if (rollout + 1) % MAX_EPILEN == 0:
					episode += 1
					# save episodic experience
					np.savetxt(fd_epilen, [len(episodic_buf)]), fd_epilen.flush()
					np.save(os.path.join(OUTLET_PATH, 'experiences/exp_{:07d}.npy'.format(episode)), episodic_buf)

					# save action entropy information
					# [0] entropy of distribution over emitted actions in one episode
					# [1, 2] maximum and minimum action entropy across rollouts in one episode
					expn = np.load(os.path.join(OUTLET_PATH, 'experiences/exp_{:07d}.npy'.format(episode)))
					dfrm = pd.DataFrame({'ai': expn[:, 6].astype('int'), 'c': 1})
					cntr = dfrm.groupby(['ai']).count()
					dist = np.zeros(121)
					dist[cntr.index] = np.array(cntr).ravel()
					dist = dist / dist.sum()
					with np.warnings.catch_warnings():
						np.warnings.filterwarnings('ignore')
						entt = np.where(dist > 0, -np.log(dist) * dist, 0).sum()
					entl = agnt.action_entropy(Variable(torch.FloatTensor(expn[:, :4])).cuda()).data.cpu().numpy()
					np.savetxt(fd_epiaen, [[entt, entl.max(), entl.min()]]), fd_epiaen.flush()

					# save mean "reward" over episode
					np.savetxt(fd_epirav, [expn[:, 8].sum() / len(expn)]), fd_epirav.flush()

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
	OUTLET_BASE = 'outlet/pgir/run-{:03d}'

	# prepare intrinsic reward samples
	IRSAMPLE = []
	for rewfile in sorted(glob('outlet/bdcl/**/history_rew.txt')):
		_rew = np.loadtxt(rewfile)
		if len(_rew) == 30000:
			IRSAMPLE.append(_rew)
	IRSAMPLE = np.array(IRSAMPLE).transpose([1, 2, 0])
	IRSAMPLE = IRSAMPLE.reshape([30000, -1]).tolist()

	# ctx = mp.get_context('spawn')
	# pl = ctx.Pool(4)
	# pl.starmap_async(main, product(range(MAX_RUN), [OUTLET_BASE]))
	# pl.close()
	# pl.join()

	for run in range(MAX_RUN):
		try:
			main(run, OUTLET_BASE, IRSAMPLE)
		except Exception as exc:
			import logging, traceback
			print('\nRun-{} failed.'.format(run + 1))
			logging.error(traceback.format_exc())
			print('-' * 80)
