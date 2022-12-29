import torch.nn as nn
from torch.autograd import Variable
import random
import torchvision.transforms.functional as TF
import math
import torch
from collections import OrderedDict
import torch.nn.functional as F
import gc
import numpy as np
from time import time
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

class SelfAdaptiveTrainingCE():
	def __init__(self, labels, num_classes=10, momentum=0.9, es=60):
		# initialize soft labels to onthot vectors
		self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
		self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
		self.momentum = momentum
		self.es = es

	def __call__(self, logits, targets, index, epoch):
		if epoch < self.es:
			return F.cross_entropy(logits, targets)

		# obtain prob, then update running avg
		prob = F.softmax(logits.detach(), dim=1)
		self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob

		# obtain weights
		weights, _ = self.soft_labels[index].max(dim=1)
		weights *= logits.shape[0] / weights.sum()

		# compute cross entropy loss, without reduction
		loss = torch.sum(-F.log_softmax(logits, dim=1) * self.soft_labels[index], dim=1)

		# sample weighted mean
		loss = (loss * weights).mean()
		return loss

def weighted_mean(x, w):
	return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
	x_bar = weighted_mean(x, w)
	s2 = weighted_mean((x - x_bar)**2, w)
	alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
	beta = alpha * (1 - x_bar) /x_bar
	return alpha, beta

class BetaMixture1D(object):
	def __init__(self, max_iters=10,
				 alphas_init=[1, 2],
				 betas_init=[2, 1],
				 weights_init=[0.5, 0.5]):
		self.alphas = np.array(alphas_init, dtype=np.float64)
		self.betas = np.array(betas_init, dtype=np.float64)
		self.weight = np.array(weights_init, dtype=np.float64)
		self.max_iters = max_iters
		self.lookup = np.zeros(100, dtype=np.float64)
		self.lookup_resolution = 100
		self.lookup_loss = np.zeros(100, dtype=np.float64)
		self.eps_nan = 1e-12

	def likelihood(self, x, y):
		return stats.beta.pdf(x, self.alphas[y], self.betas[y])

	def weighted_likelihood(self, x, y):
		return self.weight[y] * self.likelihood(x, y)

	def probability(self, x):
		return sum(self.weighted_likelihood(x, y) for y in range(2))

	def posterior(self, x, y):
		return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

	def responsibilities(self, x):
		r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
		# there are ~200 samples below that value
		r[r <= self.eps_nan] = self.eps_nan
		r /= r.sum(axis=0)
		return r

	def score_samples(self, x):
		return -np.log(self.probability(x))

	def fit(self, x):
		x = np.copy(x)

		# EM on beta distributions unsable with x == 0 or 1
		eps = 1e-4
		x[x >= 1 - eps] = 1 - eps
		x[x <= eps] = eps

		for i in range(self.max_iters):

			# E-step
			r = self.responsibilities(x)

			# M-step
			self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
			self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
			self.weight = r.sum(axis=1)
			self.weight /= self.weight.sum()

		return self

	def predict(self, x):
		return self.posterior(x, 1) > 0.5

	def create_lookup(self, y):
		x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
		lookup_t = self.posterior(x_l, y)
		lookup_t[np.argmax(lookup_t):] = lookup_t.max()
		self.lookup = lookup_t
		self.lookup_loss = x_l # I do not use this one at the end

	def look_lookup(self, x, loss_max, loss_min):
		x_i = x.clone().cpu().numpy()
		x_i = np.array((self.lookup_resolution * x_i).astype(int))
		x_i[x_i < 0] = 0
		x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
		return self.lookup[x_i]

	def plot(self):
		x = np.linspace(0, 1, 100)
		plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
		plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
		plt.plot(x, self.probability(x), lw=2, label='mixture')

	def __str__(self):
		return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)

def mixup_data(x, y, alpha=1.0):
	'''Returns mixed inputs, pairs of targets, and lambda'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	index = torch.randperm(batch_size).cuda()

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam

def mixup_data_beta(x, y, B):
	'''Returns mixed inputs, pairs of targets, and lambda'''

	batch_size = x.size()[0]
	index = torch.randperm(batch_size).cuda()

	lam = ((1 - B) + (1 - B[index]))
	mixed_x = ((1-B)/lam).unsqueeze(1).unsqueeze(2).unsqueeze(3) * x + ((1-B[index])/lam).unsqueeze(1).unsqueeze(2).unsqueeze(3) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, index

def compute_probabilities_batch(data, target, cnn_model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss):
	cnn_model.eval()
	outputs = cnn_model(data)
	outputs = F.log_softmax(outputs, dim=1)
	batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
	batch_losses.detach_()
	outputs.detach_()
	cnn_model.train()
	batch_losses = (batch_losses - bmm_model_minLoss) / (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)
	batch_losses[batch_losses >= 1] = 1-10e-4
	batch_losses[batch_losses <= 0] = 10e-4

	#B = bmm_model.posterior(batch_losses,1)
	B = bmm_model.look_lookup(batch_losses, bmm_model_maxLoss, bmm_model_minLoss)

	return torch.FloatTensor(B)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_criterion_bp(pred, y_a, y_b, lam):
	return lam * F.nll_loss(pred, y_a) + (1 - lam) * F.nll_loss(pred, y_b)

def mixup_criterion_beta(pred, y_a, y_b):
	lam = np.random.beta(32, 32)
	return lam * F.nll_loss(F.log_softmax(pred, dim=1), y_a) + (1-lam) * F.nll_loss(F.log_softmax(pred, dim=1), y_b)

def mixup_criterion_SoftHard(pred, y_a, y_b, B, index, output_x1, output_x2, Temp):
	return torch.sum(
		(0.5) * (
				(1 - B) * F.nll_loss(F.log_softmax(pred, dim=1), y_a, reduction='none') + B * (-torch.sum(F.softmax(output_x1/Temp, dim=1) * F.log_softmax(pred, dim=1), dim=1))) +
				(0.5) * (
				(1 - B[index]) * F.nll_loss(F.log_softmax(pred, dim=1), y_b, reduction='none') + B[index] * (-torch.sum(F.softmax(output_x2/Temp, dim=1) * F.log_softmax(pred, dim=1), dim=1)))) / len(
		pred)

def reg_loss_class(mean_tab,num_classes=10):
	loss = 0
	for items in mean_tab:
		loss += (1./num_classes)*torch.log((1./num_classes)/items)
	return loss

def track_training_loss(model, loadertrain):
	model.eval()

	all_losses = torch.Tensor()
	all_probs = torch.Tensor()
	all_argmaxXentropy = torch.Tensor()

	for x_train, y_train, idx in loadertrain:
		data, target = x_train.cuda(), y_train.cuda()
		prediction = model(data)
		prediction = F.log_softmax(prediction, dim=1)
		idx_loss = F.nll_loss(prediction, target, reduction = 'none')
		idx_loss.detach_()
		all_losses = torch.cat((all_losses, idx_loss.cpu()))
		probs = prediction.clone()
		probs.detach_()
		all_probs = torch.cat((all_probs, probs.cpu()))
		arg_entr = torch.max(prediction, dim=1)[1]
		arg_entr = F.nll_loss(prediction.float(), arg_entr.cuda(), reduction='none')
		arg_entr.detach_()
		all_argmaxXentropy = torch.cat((all_argmaxXentropy, arg_entr.cpu()))

	loss_tr = all_losses.data.numpy()

	# outliers detection
	max_perc = np.percentile(loss_tr, 95)
	min_perc = np.percentile(loss_tr, 5)
	loss_tr = loss_tr[(loss_tr<=max_perc) & (loss_tr>=min_perc)]

	bmm_model_maxLoss = torch.FloatTensor([max_perc]).cuda()
	bmm_model_minLoss = torch.FloatTensor([min_perc]).cuda() + 10e-6


	loss_tr = (loss_tr - bmm_model_minLoss.data.cpu().numpy()) / (bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6)

	loss_tr[loss_tr>=1] = 1-10e-4
	loss_tr[loss_tr <= 0] = 10e-4

	bmm_model = BetaMixture1D(max_iters=10)
	bmm_model.fit(loss_tr)

	bmm_model.create_lookup(1)

	return bmm_model, bmm_model_maxLoss, bmm_model_minLoss

class elr_loss(nn.Module):
	def __init__(self, num_examp, num_classes=10, beta=0.7):
		super(elr_loss, self).__init__()
		self.num_classes = num_classes
		self.target = torch.zeros(num_examp, self.num_classes).cuda()
		self.beta = beta

	def forward(self, index, output, label, lamda_elr):
		y_pred = F.softmax(output, dim=1)
		y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
		y_pred_ = y_pred.data.detach()
		self.target[index] = self.beta * self.target[index] + (1 - self.beta) * (
					(y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
		ce_loss = F.cross_entropy(output, label)
		elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
		final_loss = ce_loss + lamda_elr * elr_reg
		return final_loss

class Correcter(object):
	def __init__(self, size_of_data, num_of_classes, history_length, threshold, train_set):
		self.size_of_data = size_of_data
		self.num_of_classes = num_of_classes
		self.history_length = history_length
		self.threshold = threshold

		# prediction histories of samples
		self.all_predictions = {}
		for i in range(size_of_data):
			self.all_predictions[i] = np.zeros(history_length, dtype=int)

		# Max predictive uncertainty
		self.max_certainty = -np.log(1.0/float(self.num_of_classes))

		# Corrected label map
		self.corrected_labels = {}
		for i in range(size_of_data):
			self.corrected_labels[i] = -1

		self.update_counters = np.zeros(size_of_data, dtype=int)

		# For Logging
		self.train_set = train_set

	def async_update_prediction_matrix(self, ids, softmax_matrix):
		pred_labels = torch.argmax(softmax_matrix, dim=1).detach().cpu().numpy()
		for i in range(len(ids)):
			id = ids[i].item()
			predicted_label = np.argmax(pred_labels[i])
			# append the predicted label to the prediction matrix
			cur_index = self.update_counters[id] % self.history_length
			self.all_predictions[id][cur_index] = predicted_label
			self.update_counters[id] += 1

	def get_refurbishable_samples(self, ids):

		# check predictive uncertainty
		accumulator = {}
		for i in range(len(ids)):
			id = ids[i].item()

			predictions = self.all_predictions[id]
			accumulator.clear()

			for prediction in predictions:
				if prediction not in accumulator:
					accumulator[prediction] = 1
				else:
					accumulator[prediction] = accumulator[prediction] + 1

			p_dict = np.zeros(self.num_of_classes, dtype=float)
			for key, value in accumulator.items():
				p_dict[key] = float(value) / float(self.history_length)

			# compute predictive uncertainty
			negative_entropy = 0.0
			for i in range(len(p_dict)):
				if p_dict[i] == 0:
					negative_entropy += 0.0
				else:
					negative_entropy += p_dict[i] * np.log(p_dict[i])
			certainty = - negative_entropy / self.max_certainty

			############### correspond to the lines 12--19 of the paper ################
			# check refurbishable condition
			if certainty <= self.threshold:
				self.corrected_labels[id] = np.argmax(p_dict)
				self.train_set.targets[id] = self.corrected_labels[id]
				#########################################################################

			# reuse previously classified refurbishalbe samples
			# As we tested, this part degraded the performance marginally around 0.3%p
			# because uncertainty of the sample may afftect the performance
			elif self.corrected_labels[id] != -1:
				self.train_set.targets[id] = self.corrected_labels[id]

	def patch_clean_with_refurbishable_sample_batch(self, ids):
		# 1. separate clean and unclean samples
		# 2. get refurbishable samples
		self.get_refurbishable_samples(ids)
		# 3. merging
		return ids

	def predictions_clear(self):
		self.all_predictions.clear()
		for i in range(self.size_of_data):
			self.all_predictions[i] = np.zeros(self.history_length, dtype=int)

	def compute_new_noise_ratio(self):
		num_corrected_sample = 0
		for key, value in self.corrected_labels.items():
			if value != -1:
				num_corrected_sample += 1

		return 1.0 - float(num_corrected_sample) / float(self.size_of_data)

def lrt_correction(y_tilde, f_x, current_delta=0.3, delta_increment=0.1):
	"""
	Label correction using likelihood ratio test.
	In effect, it gradually decreases the threshold according to Algorithm 1.

	current_delta: The initial threshold $\theta$
	delta_increment: The step size, corresponding to the $\beta$ in Algorithm 1.
	"""
	corrected_count = 0
	y_noise = torch.tensor(y_tilde).clone()
	n = len(y_noise)
	f_m = f_x.max(1)[0]
	y_mle = f_x.argmax(1)
	LR = []
	for i in range(len(y_noise)):
		LR.append(float(f_x[i][int(y_noise[i])] / f_m[i]))

	for i in range(int(len(y_noise))):
		if LR[i] < current_delta:
			y_noise[i] = y_mle[i]
			corrected_count += 1

	if corrected_count < 0.001 * n:
		current_delta += delta_increment
		current_delta = min(current_delta, 0.9)

	return y_noise, current_delta

def AT(model, x, y, optimizer, args, random_start=True):
	model.eval()
	epsilon = args.eps
	num_steps = args.ns
	step_size = args.ss
	if random_start:
		x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
	else:
		x_adv = x.detach()
	for _ in range(num_steps):
		x_adv.requires_grad_()
		with torch.enable_grad():
			f_adv, logits_adv = model(x_adv, True)
			loss = F.cross_entropy(logits_adv, y)
		grad = torch.autograd.grad(loss, [x_adv])[0]
		x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
		x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
		x_adv = torch.clamp(x_adv, 0.0, 1.0)
	model.train()
	x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
	# zero gradient
	optimizer.zero_grad()
	f_adv, logits = model(x_adv, True)
	loss = F.cross_entropy(logits, y)
	return logits, loss

def trades(model, x, y, optimizer, args, random_start=True):
	model.eval()
	epsilon = args.eps
	num_steps = args.ns
	step_size = args.ss
	beta = args.beta
	if random_start:
		x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
	else:
		x_adv = x.detach()
	f, nat_output = model(x, True)
	for _ in range(num_steps):
		x_adv.requires_grad_()
		with torch.enable_grad():
			f_adv, logits_adv = model(x_adv, True)
			loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1),
							   F.softmax(nat_output, dim=1),reduction='sum')
		grad = torch.autograd.grad(loss_kl, [x_adv])[0]
		x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
		x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
		x_adv = torch.clamp(x_adv, 0.0, 1.0)
	model.train()
	x_adv = Variable(x_adv, requires_grad=False)
	optimizer.zero_grad()
	# calculate robust loss
	f, logits = model(x, True)
	f_adv, adv_logits = model(x_adv, True)
	loss_natural = F.cross_entropy(logits, y)
	loss_robust = F.kl_div(F.log_softmax(adv_logits, dim=1),
						   F.softmax(logits, dim=1),reduction='batchmean')

	loss = loss_natural + beta * loss_robust
	return adv_logits, loss

def SAT(model, x, y, index, optimizer, criterion, epoch, args, random_start=True):
	model.eval()
	epsilon = args.eps
	num_steps = args.ns
	step_size = args.ss
	if random_start:
		x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
	else:
		x_adv = x.detach()
	for _ in range(num_steps):
		x_adv.requires_grad_()
		with torch.enable_grad():
			logits_adv = model(x_adv)
			loss = F.cross_entropy(logits_adv, y)
		grad = torch.autograd.grad(loss, [x_adv])[0]
		x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
		x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
		x_adv = torch.clamp(x_adv, 0.0, 1.0)
	model.train()
	x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
	# zero gradient
	optimizer.zero_grad()
	# calculate robust loss
	logits = model(x_adv)
	loss = criterion(logits, y, index, epoch)
	return logits, loss

def SAT_TRADES(model, x, y, index, optimizer, criterion, epoch, args):
	criterion_kl = nn.KLDivLoss(size_average=False)
	batch_size = len(x)
	model.eval()
	epsilon = args.eps
	num_steps = args.ns
	step_size = args.ss
	beta = args.beta
	x_adv = x.detach() + 0.001 * torch.randn_like(x).detach()
	nat_output = model(x)
	for _ in range(num_steps):
		x_adv.requires_grad_()
		with torch.enable_grad():
			logits_adv = model(x_adv)
			loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1),
							   F.softmax(nat_output, dim=1),reduction='sum')
		grad = torch.autograd.grad(loss_kl, [x_adv])[0]
		x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
		x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
		x_adv = torch.clamp(x_adv, 0.0, 1.0)
	model.train()
	x_adv = Variable(x_adv, requires_grad=False)
	optimizer.zero_grad()
	# calculate robust loss
	f, logits = model(x, True)
	f_adv, adv_logits = model(x_adv, True)
	loss_natural = criterion(logits, y, index, epoch)
	loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))

	loss = loss_natural + beta * loss_robust
	return adv_logits, loss

class TRADES_TE():
	def __init__(self, num_samples=50000, num_classes=10, momentum=0.9, es=90, step_size=0.003, epsilon=0.031,
				 perturb_steps=10, norm='linf', beta=6.0):
		# initialize soft labels to onthot vectors
		print('number samples: ', num_samples, 'num_classes: ', num_classes)
		self.soft_labels = torch.zeros(num_samples, num_classes, dtype=torch.float).cuda(non_blocking=True)
		self.momentum = momentum
		self.es = es
		self.step_size = step_size
		self.epsilon = epsilon
		self.perturb_steps = perturb_steps
		self.norm = norm
		self.beta = beta

	def __call__(self, x_natural, y, index, epoch, model, optimizer, weight):
		criterion_kl = nn.KLDivLoss(size_average=False)
		model.eval()
		batch_size = len(x_natural)
		logits = model(x_natural)

		if epoch >= self.es:
			prob = F.softmax(logits.detach(), dim=1)
			self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob
			soft_labels_batch = self.soft_labels[index] / self.soft_labels[index].sum(1, keepdim=True)

		# generate adversarial example
		x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
		for _ in range(self.perturb_steps):
			x_adv.requires_grad_()
			with torch.enable_grad():
				logits_adv = model(x_adv)
				loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
				if epoch >= self.es:
					loss = (self.beta / batch_size) * loss_kl + weight * (
								(F.softmax(logits_adv, dim=1) - soft_labels_batch) ** 2).mean()
				else:
					loss = loss_kl
			grad = torch.autograd.grad(loss, [x_adv])[0]
			if self.norm == 'linf':
				x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
				x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
			elif self.norm == 'l2':
				g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
				scaled_grad = grad.detach() / (g_norm.detach() + 1e-10)
				x_adv = x_natural + (x_adv.detach() + self.step_size * scaled_grad - x_natural).view(x_natural.size(0),
																									 -1).renorm(p=2,
																												dim=0,
																												maxnorm=self.epsilon).view_as(
					x_natural)
			x_adv = torch.clamp(x_adv, 0.0, 1.0)

		# compute loss
		model.train()
		optimizer.zero_grad()
		x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

		# calculate robust loss
		logits = model(x_natural)
		logits_adv = model(x_adv)
		loss_natural = F.cross_entropy(logits, y)
		loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
		if epoch >= self.es:
			loss = loss_natural + self.beta * loss_robust + weight * (
						(F.softmax(logits, dim=1) - soft_labels_batch) ** 2).mean()
		else:
			loss = loss_natural + self.beta * loss_robust
		return logits_adv, loss

class PGD_TE():
	def __init__(self, num_samples=50000, num_classes=10, momentum=0.9, es=90, step_size=0.003, epsilon=0.031,
				 perturb_steps=10, norm='linf'):
		# initialize soft labels to onthot vectors
		print('number samples: ', num_samples, 'num_classes: ', num_classes)
		self.soft_labels = torch.zeros(num_samples, num_classes, dtype=torch.float).cuda(non_blocking=True)
		self.momentum = momentum
		self.es = es
		self.step_size = step_size
		self.epsilon = epsilon
		self.perturb_steps = perturb_steps
		self.norm = norm

	def __call__(self, x_natural, y, index, epoch, model, optimizer, weight):
		model.eval()
		batch_size = len(x_natural)
		logits = model(x_natural)

		if epoch >= self.es:
			prob = F.softmax(logits.detach(), dim=1)
			self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob
			soft_labels_batch = self.soft_labels[index] / self.soft_labels[index].sum(1, keepdim=True)

		# generate adversarial example
		if self.norm == 'linf':
			x_adv = x_natural.detach() + torch.FloatTensor(*x_natural.shape).uniform_(-self.epsilon,
																					  self.epsilon).cuda()
		else:
			x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
		for _ in range(self.perturb_steps):
			x_adv.requires_grad_()
			with torch.enable_grad():
				logits_adv = model(x_adv)
				if epoch >= self.es:
					loss = F.cross_entropy(logits_adv, y) + weight * (
								(F.softmax(logits_adv, dim=1) - soft_labels_batch) ** 2).mean()
				else:
					loss = F.cross_entropy(logits_adv, y)
			grad = torch.autograd.grad(loss, [x_adv])[0]
			if self.norm == 'linf':
				x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
				x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
			elif self.norm == 'l2':
				g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
				scaled_grad = grad.detach() / (g_norm.detach() + 1e-10)
				x_adv = x_natural + (x_adv.detach() + self.step_size * scaled_grad - x_natural).view(x_natural.size(0),
																									 -1).renorm(p=2,
																												dim=0,
																												maxnorm=self.epsilon).view_as(
					x_natural)
			x_adv = torch.clamp(x_adv, 0.0, 1.0)

		# compute loss
		model.train()
		optimizer.zero_grad()
		x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

		# calculate robust loss
		logits = model(x_adv)
		if epoch >= self.es:
			loss = F.cross_entropy(logits, y) + weight * ((F.softmax(logits, dim=1) - soft_labels_batch) ** 2).mean()
		else:
			loss = F.cross_entropy(logits, y)
		return logits, loss

def PENCIL(model, x, y, idx, new_y, y_all, optimizer, epoch, args, random_start=True):
	model.eval()
	epsilon = args.eps
	num_steps = args.ns
	step_size = args.ss
	if random_start:
		x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
	else:
		x_adv = x.detach()
	for _ in range(num_steps):
		x_adv.requires_grad_()
		with torch.enable_grad():
			f_adv, logits_adv = model(x_adv, True)
			loss = F.cross_entropy(logits_adv, y)
		grad = torch.autograd.grad(loss, [x_adv])[0]
		x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
		x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
		x_adv = torch.clamp(x_adv, 0.0, 1.0)
	model.train()
	x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
	# zero gradient
	optimizer.zero_grad()
	f_adv, logits = model(x_adv, True)
	logsoftmax = nn.LogSoftmax(dim=1).cuda()
	softmax = nn.Softmax(dim=1).cuda()
	if epoch < args.stage1:
		# lc is classification loss
		lc = F.cross_entropy(logits, y)
		# init y_tilde, let softmax(y_tilde) is noisy labels
		onehot = torch.zeros(y.size(0), args.num_classes).scatter_(1, y.view(-1, 1).cpu(), 10.0)
		onehot = onehot.numpy()
		new_y[idx, :] = onehot
		yy = None
	else:
		yy = y_all
		yy = yy[idx, :]
		yy = torch.FloatTensor(yy)
		yy = yy.cuda()
		yy = torch.autograd.Variable(yy, requires_grad=True)
		# obtain label distributions (y_hat)
		last_y_var = softmax(yy)
		lc = torch.mean(softmax(logits) * (logsoftmax(logits) - torch.log((last_y_var))))
		# lo is compatibility loss
		lo = F.cross_entropy(last_y_var, y)
	# le is entropy loss
	le = - torch.mean(torch.mul(softmax(logits), logsoftmax(logits)))

	if epoch < args.stage1:
		loss = lc
	elif epoch < args.stage2:
		loss = lc + args.alpha_pencil * lo + args.beta_pencil * le
	else:
		loss = lc

	return logits, loss, new_y, yy

def LabelCorrection(model, x, y, optimizer, epoch, guidedMixup_ep, bootstrap_ep_mixup, k, temp_length, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args, random_start=True):
	model.eval()
	epsilon = args.eps
	num_steps = args.ns
	step_size = args.ss
	if epoch < guidedMixup_ep:
		x, y_a, y_b, lam = mixup_data(x, y, 32)
	elif epoch < bootstrap_ep_mixup:
		if epoch == 1:
			B = 0.5 * torch.ones(len(y)).float().cuda()
		else:
			B = compute_probabilities_batch(x, y, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
			B = B.cuda()
			B[B <= 1e-4] = 1e-4
			B[B >= 1 - 1e-4] = 1 - 1e-4

		x, y_a, y_b, index = mixup_data_beta(x, y, B)
	else:
		output_x1 = model(x)
		output_x1.detach_()
		if epoch == 1:
			B = 0.5 * torch.ones(len(y)).float().cuda()
		else:
			B = compute_probabilities_batch(x, y, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
			B = B.cuda()
			B[B <= 1e-4] = 1e-4
			B[B >= 1 - 1e-4] = 1 - 1e-4

		temp_vec = np.linspace(1, 0.001, temp_length)
		x, y_a, y_b, index = mixup_data_beta(x, y, B)
	if random_start:
		x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
	else:
		x_adv = x.detach()
	for _ in range(num_steps):
		x_adv.requires_grad_()
		with torch.enable_grad():
			f_adv, logits_adv = model(x_adv, True)
			if epoch < guidedMixup_ep:
				criterion = nn.CrossEntropyLoss()
				loss = mixup_criterion(criterion, logits_adv, y_a, y_b, lam)
			elif epoch < bootstrap_ep_mixup:
				loss = mixup_criterion_beta(logits_adv, y_a, y_b)
			else:
				loss = mixup_criterion_beta(logits_adv, y_a, y_b)
		grad = torch.autograd.grad(loss, [x_adv])[0]
		x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
		x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
		x_adv = torch.clamp(x_adv, 0.0, 1.0)
	model.train()
	x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
	# zero gradient
	optimizer.zero_grad()
	f_adv, logits = model(x_adv, True)
	if epoch < guidedMixup_ep:
		output = F.log_softmax(logits, dim=1)
		loss = mixup_criterion_bp(output, y_a, y_b, lam)
	elif epoch < bootstrap_ep_mixup:
		loss = mixup_criterion_beta(logits, y_a, y_b)
	else:
		output_mean = F.softmax(logits, dim=1)
		output = F.log_softmax(logits, dim=1)
		output_x2 = output_x1[index, :]
		tab_mean_class = torch.mean(output_mean, -2)
		Temp = temp_vec[k]
		loss = mixup_criterion_SoftHard(output, y_a, y_b, B, index, output_x1, output_x2, Temp)
		loss_reg = reg_loss_class(tab_mean_class, args.num_classes)
		loss = loss + 1.0 * loss_reg
	return logits, loss

def ELR(model, x, y, idx, epoch, lamda_elr, criterion, optimizer, args, random_start=True):
	model.eval()
	epsilon = args.eps
	num_steps = args.ns
	step_size = args.ss
	if random_start:
		x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
	else:
		x_adv = x.detach()
	for _ in range(num_steps):
		x_adv.requires_grad_()
		with torch.enable_grad():
			f_adv, logits_adv = model(x_adv, True)
			if epoch > 0:
				loss = F.cross_entropy(logits_adv, torch.argmax(criterion.target[idx], dim=1))
			else:
				loss = F.cross_entropy(logits_adv, y)
		grad = torch.autograd.grad(loss, [x_adv])[0]
		x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
		x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
		x_adv = torch.clamp(x_adv, 0.0, 1.0)
	model.train()
	x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
	# zero gradient
	optimizer.zero_grad()
	f_adv, logits = model(x_adv, True)
	loss = criterion(idx, logits, y, lamda_elr)
	return logits, loss

def SELFIE(model, x, y, optimizer, args, random_start=True):
	model.eval()
	epsilon = args.eps
	num_steps = args.ns
	step_size = args.ss
	if random_start:
		x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
	else:
		x_adv = x.detach()
	for _ in range(num_steps):
		x_adv.requires_grad_()
		with torch.enable_grad():
			f_adv, logits_adv = model(x_adv, True)
			loss = F.cross_entropy(logits_adv, y)
		grad = torch.autograd.grad(loss, [x_adv])[0]
		x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
		x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
		x_adv = torch.clamp(x_adv, 0.0, 1.0)
	model.train()
	x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
	# zero gradient
	optimizer.zero_grad()
	f_adv, logits = model(x_adv, True)
	loss = F.cross_entropy(logits, y)
	return logits, loss

def PLC(model, x, y, optimizer, args, random_start=True):
	model.eval()
	epsilon = args.eps
	num_steps = args.ns
	step_size = args.ss
	if random_start:
		x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
	else:
		x_adv = x.detach()
	for _ in range(num_steps):
		x_adv.requires_grad_()
		with torch.enable_grad():
			f_adv, logits_adv = model(x_adv, True)
			loss = F.cross_entropy(logits_adv, y)
		grad = torch.autograd.grad(loss, [x_adv])[0]
		x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
		x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
		x_adv = torch.clamp(x_adv, 0.0, 1.0)
	model.train()
	x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
	# zero gradient
	optimizer.zero_grad()
	f_adv, logits = model(x_adv, True)
	loss = F.cross_entropy(logits, y)
	return logits, loss



