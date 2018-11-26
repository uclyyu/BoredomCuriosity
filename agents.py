import torch, copy, math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import grad, Variable, Function
from collections import deque
from itertools import chain
from torchmate.linalg.batched_op import bminv, bchol
from torchmate.nn.functional import sthru


class GradientBridge(Function):
	@staticmethod
	def forward(ctx, x, y):
		return x
	@staticmethod
	def backward(ctx, grad_input):
		return None, grad_input
	
gbridge = GradientBridge.apply

_action_space = np.mgrid[-2:2:11j, -2:2:11j] 
_space_size = 121

class UpdateControl:
	def __init__(self, fd, param, lr, scheduler_args={'mode': 'min', 'factor': 0.1, 'patience': 10, 'verbose': True}, ignore=False):
		self.fd = fd
		self.counter = 0
		self.total_loss = 0
		self.adam = optim.Adam(param, lr)
		self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.adam, **scheduler_args)
		self.ignore = ignore
	def step(self):
		self.adam.step()
		self.adam.zero_grad()
	def register_loss(self, l):
		if np.isnan(float(l)):
			raise ValueError('loss value is NaN.')
		self.total_loss += float(l)
		self.counter += 1
	def ignore_scheduler(self, b):
		self.ignore = bool(b)
	def lr_step(self):
		mean_loss = self.total_loss / self.counter
		np.savetxt(self.fd, [mean_loss]), self.fd.flush()
		self.counter = 0
		self.total_loss = 0
		if not self.ignore:
			self.scheduler.step(mean_loss)


class AgentBASELINE(nn.Module):
	def __init__(self):
		super(AgentBASELINE, self).__init__()
		# forward model
		self.fm = nn.ModuleList([
			nn.Sequential(
				nn.Linear(  4, 512), nn.ReLU(True),
				nn.Linear(512, 512), nn.ReLU(True),
				nn.Linear(512, 512), nn.ReLU(True),
				nn.Linear(512,   4)),
			nn.Linear(4, 512),
			nn.Sequential(
				nn.Linear(  4, 512), nn.ReLU(True),
				nn.Linear(512, 512), nn.ReLU(True),
				nn.Linear(512, 512), nn.ReLU(True),
				nn.Linear(512, 512, bias=False)),
			nn.ReLU(True),
			nn.Linear(512, 16, bias=False),  # "A"
			nn.Linear(512, 32, bias=False),  # "B"
			nn.Linear(512,  8, bias=False),  # "C"
			nn.Linear(512,  1, bias=False)   # "o"
		])
	
	def eval_fm(self, s, a):
		# evaluate forward model
		s_ = self.fm[0](s)
		s0 = s - s_
		hid = self.fm[3](self.fm[1](s0) + self.fm[2](s0))
		A = self.fm[4](hid).view(-1, 4, 4)
		B = self.fm[5](hid).view(-1, 4, 4, 2)
		C = self.fm[6](hid).view(-1, 2, 4)
		o = self.fm[7](hid)
		J = A + a.view(-1, 1, 1, 2).mul(B).sum(3)
		mean = (s0 + s_).view(-1, 4, 1).mul(J).sum(1) + a.view(-1, 2, 1).mul(C).sum(1) + o
		cov = J.view(-1, 4, 4, 1).mul(J.view(-1, 4, 1, 4)).sum(1)
		return mean, cov
	
	def forward(self, s, a):
		return self.eval_fm(s, a)[0]


class AgentAUDP(AgentBASELINE):
	def __init__(self):
		super(AgentAUDP, self).__init__()
	
	def forward(self, s, a):
		return self.eval_fm(s, a)


class AgentPGUR(AgentAUDP):
	def __init__(self, action_space=_action_space, space_size=_space_size):
		super(AgentPGUR, self).__init__()
		self.register_buffer('action_space', torch.FloatTensor(action_space).view(2, -1).t())
		# action policy
		self.ap = nn.ModuleList([
			nn.Linear(4, 512),
			nn.Sequential(
				nn.Linear(  4, 512), nn.ELU(True),
				nn.Linear(512, 512), nn.ELU(True),
				nn.Linear(512, 512), nn.ELU(True),
				nn.Linear(512, 512, bias=False)),
			nn.Sequential(
				nn.ELU(True),
				nn.Linear(512, space_size))
		])

	def action_entropy(self, s):
		logit = self.ap[2](self.ap[0](s) + self.ap[1](s))
		return F.softmax(logit, dim=1).mul(-F.log_softmax(logit, dim=1)).sum(1, keepdim=True)
		
	def query_action(self, s, index=None, n=1):
		logit = self.ap[2](self.ap[0](s) + self.ap[1](s))
		prob = F.softmax(logit, dim=1)
		if index is None:
			index = torch.multinomial(prob, n, replacement=True)
			a = Variable(self.action_space.gather(0, index.data.expand(-1, 2)))
			p = prob.gather(1, index)
			return gbridge(a, p.expand(-1, 2)), index, p
		else:
			a = Variable(self.action_space.gather(0, index.data.expand(-1, 2)))
			p = prob.gather(1, index)
			return gbridge(a, p.expand(-1, 2)), index, p

	def infer_action_prob(self, s, a_index):
		logit = self.ap[2](self.ap[0](s) + self.ap[1](s))
		prob = F.softmax(logit, dim=1)
		p = prob.gather(1, a_index)
		return p
	
	def forward(self, s, a):
		return self.eval_fm(s, a)


class AgentPEDL(AgentPGUR):
	def __init__(self, **kwargs):
		super(AgentPEDL, self).__init__(**kwargs)
		# value function
		self.vf = nn.ModuleList([
			nn.Linear(4, 512),
			nn.Sequential(
				nn.Linear(  4, 512), nn.ELU(True),
				nn.Linear(512, 512), nn.ELU(True),
				nn.Linear(512, 512), nn.ELU(True),
				nn.Linear(512, 512, bias=False)),
			nn.Sequential(
				nn.ELU(True),
				nn.Linear(512, 1))
		])
	
	def eval_vf(self, s, s_=None):
		# evaluate value function
		if s_ is not None:
			s = gbridge(s, s_)
		return self.vf[2](self.vf[0](s) + self.vf[1](s))
		
	def eval_rf(self, s0, a0, s1):
		return (self.eval_fm(s0, a0)[0] - s1).pow(2).sum(1, keepdim=True)
	
	def forward(self, s, a):
		return self.eval_fm(s, a)


class AgentBDCL(AgentPEDL):
	def __init__(self, **kwargs):
		super(AgentBDCL, self).__init__(**kwargs)
		self.register_buffer('eye', torch.eye(4).view(1, 4, 4))
		self.register_buffer('eye_mask', torch.eye(4).byte().unsqueeze(0))
		
		# marginalised model
		self.mm = nn.ModuleList([
			# ... mean
			nn.Linear(4, 512),
			nn.Sequential(
				nn.Linear(  4, 512), nn.ELU(True),
				nn.Linear(512, 512), nn.ELU(True),
				nn.Linear(512, 512, bias=False)),
			nn.Sequential(
				nn.ELU(True), 
				nn.Linear(512, 4)),
			# ... Householder vector
			nn.Linear(4, 512),
			nn.Sequential(
				nn.Linear(  4, 512), nn.ELU(True),
				nn.Linear(512, 512), nn.ELU(True),
				nn.Linear(512, 512, bias=False)),
			nn.Sequential(
				nn.ELU(True), 
				nn.Linear(512, 4)),
			# ... positive diagonal terms
			nn.Linear(4, 512),
			nn.Sequential(
				nn.Linear(  4, 512), nn.ELU(True),
				nn.Linear(512, 512), nn.ELU(True),
				nn.Linear(512, 512, bias=False)),
			nn.Sequential(
				nn.ELU(True), 
				nn.Linear(512, 4),
				nn.Softplus())
		])
	
	def eval_mm(self, s):
		# evaluate marginalised forward model
		I = Variable(self.eye)
		s_ = self.fm[0](s)
		s0 = s - s_
		mean = self.mm[2](self.mm[0](s0) + self.mm[1](s0))
		v = self.mm[5](self.mm[3](s0) + self.mm[4](s0))
		d = self.mm[8](self.mm[6](s0) + self.mm[7](s0)).view(-1, 1, 4)
		H = I - 2 * v.unsqueeze(2).mul(v.unsqueeze(1)).div(v.pow(2).sum(1).view(-1, 1, 1))
		cov = torch.bmm(H.mul(d), H)
		return mean, cov
		
	def eval_rf(self, f_mean, f_cov, e_mean, e_cov):
		fbs = f_mean.size(0)
		ebs = e_mean.size(0)
		mask = self.eye_mask
		# evaluate "reward" function
		# ... cholesky factors
		f_chr = bchol(f_cov)  # [m.potrf() for m in torch.unbind(f_cov, dim=0)]
		e_chr = bchol(e_cov)  # [m.potrf() for m in torch.unbind(e_cov, dim=0)]
		# ... matrix inversion (potri not differentiable)
		e_inv = bminv(e_cov)  # torch.stack([m.inverse() for m in torch.unbind(e_cov, dim=0)], dim=0)  # e_inv = torch.stack([m.potri() for m in e_chr], dim=0)
		# ... log matrix determinant
		f_ldt = f_chr[mask].view(fbs, -1).log().sum(1, keepdim=True) * 2  # torch.stack([2 * m.diag().log().sum() for m in f_chr], dim=0)
		e_ldt = e_chr[mask].view(ebs, -1).log().sum(1, keepdim=True) * 2  # torch.stack([2 * m.diag().log().sum() for m in e_chr], dim=0)
		# ... error
		error = f_mean - e_mean
		r = (
			torch.bmm(e_inv, f_cov)[mask].view(-1, 4).sum(1, keepdim=True) +
			error.unsqueeze(2).mul(e_inv).sum(1).mul(error).sum(1, keepdim=True) - 4 +
			e_ldt - f_ldt
		)
		return 0.5 * r
	
	def forward(self, s, a):
		return self.eval_fm(s, a)


class AgentBDCLGAUS(AgentBDCL):
	def __init__(self, **kwargs):
		super().__init__()
		self.register_buffer('eta_proto', torch.FloatTensor([0]))
		self.ap = nn.ModuleList([
			nn.Linear(4, 512),
			nn.Sequential(
				nn.Linear(  4, 512), nn.ELU(True),
				nn.Linear(512, 512), nn.ELU(True),
				nn.Linear(512, 512), nn.ELU(True),
				nn.Linear(512, 512, bias=False)),
			nn.Sequential(
				nn.ELU(True),
				nn.Linear(512, 2)),
			nn.Sequential(
				nn.ELU(True),
				nn.Linear(512, 2))
		])

	def forward(self, s, a):
		return self.eval_fm(s, a)

	def gaus_action_entropy(self, s):
		hid = self.ap[0](s) + self.ap[1](s)
		logv = torch.clamp(self.ap[3](hid), -2, 2)
		return logv.sum(1, keepdim=True)

	def gaus_query_action(self, s, a=None):
		hid = self.ap[0](s) + self.ap[1](s)
		mean = torch.sin(self.ap[2](hid)) 
		logv = torch.clamp(self.ap[3](hid), -2, 2)
		if a is None:
			eta = Variable(self.eta_proto.new(logv.size()).normal_())
			act = mean + (0.5 * logv).exp() * eta
		else:
			eta = ((a - mean) / (0.5 * logv).exp()).detach()
			act = mean + (0.5 * logv).exp() * eta
		lpr = -0.5 * (2 * math.log(2 * math.pi) + logv.sum(1, keepdim=True) + (mean - act).pow(2).div(logv.exp()).sum(1, keepdim=True))
		return mean, lpr

	def gaus_infer_action_lpr(self, s, a):
		hid = self.ap[0](s) + self.ap[1](s)
		mean = torch.sin(self.ap[2](hid))
		logv = torch.clamp(self.ap[3](hid), -2, 2)
		lpr = -0.5 * (2 * math.log(2 * math.pi) + logv.sum(1, keepdim=True) + (mean - a).pow(2).div(logv.exp()).sum(1, keepdim=True))
		return lpr
