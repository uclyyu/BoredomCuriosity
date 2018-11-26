# bdcl, boredom-driven curious learning
# Setup-03
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

def copy_vf(src, dst):
	for ps, pd in zip(src.vf.parameters(), dst.vf.parameters()):
		pd.data = ps.data.clone()

def copy_mm(src, dst):
	for ps, pd in zip(src.mm.parameters(), dst.mm.parameters()):
		pd.data = ps.data.clone()

def copy_fm(src, dst):
	for ps, pd in zip(src.fm.parameters(), dst.fm.parameters()):
		pd.data = ps.data.clone()	
	
def _sampleDB(dblist, n, cuda=False):
	samp = np.array(random.sample(dblist, n))
	s0, a0, i0, p0, s1, gate = map(lambda o: Variable(torch.from_numpy(o).float()), np.split(samp, [4, 6, 7, 8, 12], axis=1))
	i0 = i0.long()
	if cuda:
		s0, a0, i0, p0, s1, gate = map(lambda o: o.cuda(), [s0, a0, i0, p0, s1, gate])
	return s0, a0, i0, p0, s1, gate

def fit_fm(roll, agnt, agnt_clone, ctl_fm, dblist, n, cuda=False):
	# 1. update fm
	copy_fm(agnt, agnt_clone)
	s0, a0, i0, p0, s1, gate = _sampleDB(dblist, n, cuda=cuda)  # s0, a0, i0, p0, s1, gate, j0 = _sampleDB(dblist, n, cuda=cuda)
	fm_loss = (agnt.eval_fm(s0, a0)[0] - s1).pow(2).sum(1).mean()
	ctl_fm.register_loss(fm_loss)
	for p, g in zip(agnt.fm.parameters(), grad(fm_loss, agnt.fm.parameters())):
		if p.grad is None:
			p.grad = g
		else:
			p.grad += g
	ctl_fm.step()
	if (roll + 1) % 300 == 0:
		ctl_fm.lr_step()
	
def fit_vf(roll, agnt, agnt_clone, ctl_vf, dblist, n, gamma, repeat=1, cuda=False):
	# 2. update vf
	copy_vf(agnt, agnt_clone)
	for rep in range(repeat):
		s0, a0, i0, p0, s1, gate = _sampleDB(dblist, n, cuda=cuda)
		w1, c1 = agnt.eval_fm(s0, a0)
		r1cur = agnt.eval_rf(w1, c1, *agnt.eval_mm(s0))
		r1pst = agnt_clone.eval_rf(w1, c1, *agnt_clone.eval_mm(s0))
		r1 = r1pst - r1cur
		a_ratio = torch.clamp(agnt.infer_action_prob(s0, i0) / p0, 0, 100)
		v0 = gate * (r1 + gamma * agnt_clone.eval_vf(s1))
		vf_loss = (0.5 * a_ratio * (- agnt.eval_vf(s0) + v0).pow(2)).mean()
		ctl_vf.register_loss(vf_loss)
		
		for p, g in zip(agnt.vf.parameters(), grad(vf_loss, agnt.vf.parameters())):
			if p.grad is None:
				p.grad = g
			else:
				p.grad += g
		ctl_vf.step()
		if (rep + 1) % 50 == 0:
			copy_vf(agnt, agnt_clone)

	if (roll + 1) % 300 == 0:
		ctl_vf.lr_step()
	
def fit_ap(roll, agnt, agnt_clone, ctl_ap, dblist, n, gamma, cuda=False, samp=[], aux=[]):
	# 3. update ap
	if len(samp) == 0:
		samp.extend(_sampleDB(dblist, n, cuda=cuda))
	s0, _, i0, p0, s1, gate = samp
	a0, _, q0 = agnt.query_action(s0, i0)
	w1, c1 = agnt.eval_fm(s0, a0)
	r1cur = agnt.eval_rf(w1, c1, *agnt.eval_mm(s0))
	r1pst = agnt_clone.eval_rf(w1, c1, *agnt_clone.eval_mm(s0))
	r1 = r1pst - r1cur
	# rev = 2 * (r1 > 0).detach().float()
	v0 = gate * r1 + gamma * agnt.eval_vf(s1, w1)
	a_ratio = (q0 / p0)
	clip_ratio = torch.clamp(a_ratio, 0.8, 1.2)
	ap_loss = - torch.min(a_ratio * v0, clip_ratio * v0).sum()
	ctl_ap.register_loss(ap_loss)
	for p, g in zip(agnt.ap.parameters(), grad(ap_loss, agnt.ap.parameters())):
		if p.grad is None:
			p.grad = g
		else:
			p.grad += g
	ctl_ap.step()
	aux.extend(r1.data.cpu().numpy().ravel())

	if (roll + 1) % 300 == 0:
		ctl_ap.lr_step()


def fit_mm(roll, agnt, agnt_clone, ctl_mm, dblist, n, cuda=False, samp=[]):
	# 4. update mm
	copy_mm(agnt, agnt_clone)
	if len(samp) == 0:
		samp.extend(_sampleDB(dblist, n, cuda=cuda))
	s0, a0, i0, p0, s1, gate = samp
	w1, c1 = agnt.eval_fm(s0, a0)
	bore_rate = torch.clamp(
		((s1 -                            w1).pow(2).sum(1, keepdim=True).mul(-1) - 
		 (s1 - agnt_clone.eval_fm(s0, a0)[0]).pow(2).sum(1, keepdim=True).mul(-1)).exp(),
		 0, 100)
	mm_loss = bore_rate * agnt.eval_rf(w1, c1, *agnt.eval_mm(s0)).mean()
	mm_loss = agnt.eval_rf(w1, c1, *agnt.eval_mm(s0)).mean()
	ctl_mm.register_loss(mm_loss)
	for p, g in zip(agnt.mm.parameters(), grad(mm_loss, agnt.mm.parameters())):
		if p.grad is None:
			p.grad = g
		else:
			p.grad += g
	ctl_mm.step()
	if (roll + 1) % 300 == 0:
		ctl_mm.lr_step()

def post_fit_fm(roll, agnt, ctl, dblist, n, cuda=False):
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

def eval_vf(agnt, s, size=(50, 50)):
	v = agnt.eval_vf(s).view(*size, -1).data.cpu().numpy()
	vmean = v.mean(axis=2).T[::-1]
	vstd = np.std(v, axis=2).T[::-1]
	vt = vmean / vstd
	return [vmean, vstd, vt]

def main(run, ob, sg):
	LR_FM = 5e-5  # world-model larning rate
	LR_VF = 5e-5  # value function learning rate
	LR_AP = 5e-5  # action policy learning rate
	LR_MM = 5e-5  # meta-model learning rate
	GAMMA = 0.999 # discount factor
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
		os.mkdir(os.path.join(OUTLET_PATH, 'values'))
		os.mkdir(os.path.join(OUTLET_PATH, 'experiences'))

	# starts a new run with new agent
	agnt = agents.AgentBDCL().cuda()
	agnt_clone = copy.deepcopy(agnt)

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

	# start rollout and post-rollout
	with open(os.path.join(OUTLET_PATH, 'fm_loss.txt'), 'wb') as fd_fm_loss, \
		 open(os.path.join(OUTLET_PATH, 'mm_loss.txt'), 'wb') as fd_mm_loss, \
		 open(os.path.join(OUTLET_PATH, 'vf_loss.txt'), 'wb') as fd_vf_loss, \
		 open(os.path.join(OUTLET_PATH, 'ap_loss.txt'), 'wb') as fd_ap_loss, \
		 open(os.path.join(OUTLET_PATH, 'episode_aen.txt'), 'wb') as fd_epiaen, \
		 open(os.path.join(OUTLET_PATH, 'episode_len.txt'), 'wb') as fd_epilen, \
		 open(os.path.join(OUTLET_PATH, 'history_rew.txt'), 'wb') as fd_rewhis, \
		 tqdm(range(N_PREROLL)) as pre_pbar, \
		 tqdm(range(N_PREROLL, N_POSTROLL + N_PREROLL)) as post_pbar:
		
			ctl_fm = agents.UpdateControl(fd_fm_loss, agnt.fm.parameters(), LR_FM, ignore=True)
			ctl_mm = agents.UpdateControl(fd_mm_loss, agnt.mm.parameters(), LR_MM, ignore=True)
			ctl_vf = agents.UpdateControl(fd_vf_loss, agnt.vf.parameters(), LR_VF, ignore=True)
			ctl_ap = agents.UpdateControl(fd_ap_loss, agnt.ap.parameters(), LR_AP, ignore=True)
			dbl = db2list(experienceDB)
			env = SpaceShip()
			episodic_buf = []
			episode = 0
			experienceDB.append(episodic_buf)
			
			for rollout in pre_pbar:
				# auxiliary list for collecting intrinsic rewards
				rewaux = []
				samaux = []
				# step and fill episodic memory
				if step_fill(env, agnt, episodic_buf, cuda=True):
					pre_pbar.set_description('=Run-{}=PreRoll-{}.Go='.format(run + 1, rollout + 1))
				else:
					pre_pbar.set_description('=Run-{}=PreRoll-{}.Nogo='.format(run + 1, rollout + 1))

				# save model on rollout
				if (rollout + 1) == 1 or (rollout + 1) % SAVE_INTERVAL == 0:
					torch.save(agnt.state_dict(), os.path.join(OUTLET_PATH, 'models/agnt_{:07d}.pt'.format(rollout + 1)))
				
				# fit model on rollout and save values
				dbl = db2list(experienceDB)
				if rollout == 0 or (rollout + 1) % MAX_EPILEN == 0:
					fit_vf(rollout, agnt, agnt_clone, ctl_vf, dbl, SAMPLE_SIZE, GAMMA, repeat=2000, cuda=True)
				else:
					fit_vf(rollout, agnt, agnt_clone, ctl_vf, dbl, SAMPLE_SIZE, GAMMA, repeat=1, cuda=True)
				fit_mm(rollout, agnt, agnt_clone, ctl_mm, dbl, SAMPLE_SIZE,        cuda=True, samp=samaux)
				fit_ap(rollout, agnt, agnt_clone, ctl_ap, dbl, SAMPLE_SIZE, GAMMA, cuda=True, samp=samaux, aux=rewaux)
				fit_fm(rollout, agnt, agnt_clone, ctl_fm, dbl, SAMPLE_SIZE,        cuda=True)

				# save reward history
				np.savetxt(fd_rewhis, [rewaux]), fd_rewhis.flush()

				# collect episodic information
				if (rollout + 1) % MAX_EPILEN == 0:
					episode += 1
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
					
					if (rollout + 1) < N_PREROLL:
						# new episodic buffer
						episodic_buf = []
						experienceDB.append(episodic_buf)

			# Train world model after accumulating experience
			dbl = []
			for file in glob(os.path.join(OUTLET_PATH, 'experiences/exp_*.npy')):
				dbl.append(np.load(file))
			dbl = list(chain(*dbl))

			post_pbar.set_description('=Run-{}=PostData Phase='.format(run + 1))
			ctl_fm.ignore_scheduler(False)
			for rollout in post_pbar:
				if (rollout + 1) % SAVE_INTERVAL == 0:
					torch.save(agnt.state_dict(), os.path.join(OUTLET_PATH, 'models/agnt_{:07d}.pt'.format(rollout + 1)))
				post_fit_fm(rollout, agnt, ctl_fm, dbl, SAMPLE_SIZE, cuda=True)


if __name__ == '__main__':
	MAX_RUN = 128
	OUTLET_BASE = 'outlet/bdcl/run-{:03d}'
	# state grid for evaluating value function approximator
	# STATE_GRID = Variable(torch.FloatTensor(np.mgrid[0:1:50j, 0:1:50j, -1:1:10j, -1:1:10j].reshape([4, -1]).T)).cuda()

	for run in range(MAX_RUN):
		try:
			main(run, OUTLET_BASE, None)
		except Exception as exc:
			import logging, traceback
			print('\nRun-{} failed.'.format(run + 1))
			logging.error(traceback.format_exc())
			print('-' * 80)
