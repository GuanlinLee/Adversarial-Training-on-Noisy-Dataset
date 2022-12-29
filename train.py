import copy
import torchvision
import torchvision.transforms as transforms
import torch
import torch.utils.data
import resnet
from torch import nn
import torchattacks
from tqdm import tqdm
import logging
import random
import os,sys
import numpy as np
import argparse
import dataloader
import wrn
import loss_functions
from torch.optim.lr_scheduler import MultiStepLR
import time
from datetime import timedelta
import math
from logging import getLogger
from PIL import ImageFilter
from torch.nn import functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
					help='model architecture')
parser.add_argument('--dataset', default='cifar10', type=str,
					help='which dataset used to train')
parser.add_argument('--num_classes', default=10, type=int, metavar='N',
					help='number of classes')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='wd')
parser.add_argument('--save', default='M2.pkl', type=str,
					help='model save name')
parser.add_argument('--seed', type=int,
					default=0, help='random seed')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--eps', type=float, default=8./255., help='perturbation bound')
parser.add_argument('--ns', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--ss', type=float, default=2./255., help='step size')
parser.add_argument('--beta', type=float, default=6.0)


parser.add_argument('--exp', default='exp_test', type=str,
					help='exp name')
parser.add_argument('--method', default='pgd', type=str,
					help='AT method to use')
parser.add_argument('--nr', type=float, default=0.2,
					help='noisy ratio for dataset')
parser.add_argument('--noise_type', type=str, default='sym',
					help='type of label noise')

#SAT Settings
parser.add_argument('--ES', type=int, default=60)

#TE Settings
parser.add_argument('--te-alpha', default=0.9, type=float,
					help='momentum term of self-adaptive training')
parser.add_argument('--start-es', default=90, type=int,
					help='start epoch of self-adaptive training (default 0)')
parser.add_argument('--end-es', default=150, type=int,
					help='start epoch of self-adaptive training (default 0)')
parser.add_argument('--reg-weight', default=300, type=float)

#PENCIL Settings
parser.add_argument('--alpha_pencil', default=0.4, type=float,
                    metavar='H-P', help='the coefficient of Compatibility Loss')
parser.add_argument('--beta_pencil', default=0.1, type=float,
                    metavar='H-P', help='the coefficient of Entropy Loss')
parser.add_argument('--lambda1', default=600, type=int,
                    metavar='H-P', help='the value of lambda')
parser.add_argument('--stage1', default=44, type=int,
                    metavar='H-P', help='number of epochs utill stage1')
parser.add_argument('--stage2', default=125, type=int,
                    metavar='H-P', help='number of epochs utill stage2')
args = parser.parse_args()

if args.dataset == 'cifar10':
	args.num_classes = 10
else:
	args.num_classes = 100

class LogFormatter:
	def __init__(self):
		self.start_time = time.time()

	def format(self, record):
		elapsed_seconds = round(record.created - self.start_time)

		prefix = "%s - %s - %s" % (
			record.levelname,
			time.strftime("%x %X"),
			timedelta(seconds=elapsed_seconds),
		)
		message = record.getMessage()
		message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
		return "%s - %s" % (prefix, message) if message else ""
def create_logger(filepath, rank):
	# create log formatter
	log_formatter = LogFormatter()

	# create file handler and set level to debug
	if filepath is not None:
		if rank > 0:
			filepath = "%s-%i" % (filepath, rank)
		file_handler = logging.FileHandler(filepath, "a")
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(log_formatter)

	# create console handler and set level to info
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(log_formatter)

	# create logger and set level to debug
	logger = logging.getLogger()
	logger.handlers = []
	logger.setLevel(logging.DEBUG)
	logger.propagate = False
	if filepath is not None:
		logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	# reset logger elapsed time
	def reset_time():
		log_formatter.start_time = time.time()

	logger.reset_time = reset_time

	return logger
def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)

def sigmoid_rampup(current, start_es, end_es):
	"""Exponential rampup from https://arxiv.org/abs/1610.02242"""
	if current < start_es:
		return 0.0
	if current > end_es:
		return 1.0
	else:
		import math
		phase = 1.0 - (current - start_es) / (end_es - start_es)
		return math.exp(-5.0 * phase * phase)


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
setup_seed(args.seed)


logger = getLogger()
if not os.path.exists(args.dataset+'/'+ args.arch +'/'+args.exp):
	os.makedirs(args.dataset+'/'+ args.arch +'/'+args.exp)
logger = create_logger(
	os.path.join(args.dataset+'/'+ args.arch +'/'+args.exp + '/', args.exp + ".log"), rank=0
)
logger.info("============ Initialized logger ============")
logger.info(
	"\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
)
args.save = args.dataset+'/'+ args.arch +'/'+args.exp + '/' +  args.save


wd=args.wd
learning_rate=args.lr
epochs=args.epochs
batch_size=args.batch_size
torch.backends.cudnn.benchmark = True

transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
							  transforms.RandomHorizontalFlip(),
							  torchvision.transforms.ToTensor(),
							  ])
transform_test=transforms.Compose([torchvision.transforms.Resize((32,32)),
								   transforms.ToTensor(),
								   ])



trainset, testset = dataloader.get_dataset(args, transform, transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
										  shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
										 shuffle=False, num_workers=4)
num_classes = trainset.num_classes
targets = np.asarray(trainset.targets)
if args.arch == 'resnet':
	n = resnet.resnet18(args.dataset).cuda()
elif args.arch == 'wrn':
	n = wrn.WideResNet(num_classes=args.num_classes).cuda()


optimizer = torch.optim.SGD(n.parameters(),momentum=args.momentum,
							lr=learning_rate,weight_decay=wd)


milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]

scheduler = MultiStepLR(optimizer,milestones=milestones,gamma=args.gamma)

if args.method == 'pgd_te':
	pgd_te = loss_functions.PGD_TE(num_samples=50000,
					   num_classes=args.num_classes,
					   momentum=args.te_alpha,
					   step_size=args.ss,
					   epsilon=args.eps,
					   perturb_steps=args.ns,
					   norm='linf',
					   es=args.start_es)
elif args.method == 'trades_te':
	trades_te = loss_functions.TRADES_TE(num_samples=50000,
						  num_classes=args.num_classes,
						  momentum=args.te_alpha,
						  step_size=args.ss,
						  epsilon=args.eps,
						  perturb_steps=args.ns,
						  norm='linf',
						  es=args.start_es,
						  beta=args.beta)

if 'sat' in args.method:
	criterion = loss_functions.SelfAdaptiveTrainingCE(labels=targets, num_classes=num_classes, es=args.ES)

if args.method == 'labelcorr':
	guidedMixup_ep = 71
	bootstrap_ep_mixup = guidedMixup_ep + 5
	temp_length = 133 - bootstrap_ep_mixup
	bmm_model = bmm_model_maxLoss = bmm_model_minLoss = k = 0

if args.method == 'elr':
	if args.dataset == 'cifar10':
		if args.noise_type == 'sym':
			criterion = loss_functions.elr_loss(50000, num_classes=10, beta=0.7)
			lambda_elr = 3
		else:
			criterion = loss_functions.elr_loss(50000, num_classes=10, beta=0.9)
			lambda_elr = 1
	else:
		criterion = loss_functions.elr_loss(50000, num_classes=100, beta=0.9)
		lambda_elr = 7

if args.method == 'selfie':
	warm_up = 25
	threshold = 0.05
	queue_size = 15
	correcter = loss_functions.Correcter(50000, num_classes, queue_size, threshold, copy.deepcopy(trainset))
	loss_judge = nn.CrossEntropyLoss(reduction='none')

if args.method == 'plc':
	rollwindow = 5
	warm_up = 8
	inc = 0.1
	current_delta = 0.3
	f_record = torch.zeros([rollwindow, 50000, num_classes])


train_clean_acc = []
train_adv_acc = []
test_clean_acc = []
test_adv_acc = []
best_eval_acc = 0.0

for epoch in range(epochs):
	rampup_rate = sigmoid_rampup(epoch+1, args.start_es, args.end_es)
	weight = rampup_rate * args.reg_weight
	if args.method == 'pencil':
		new_y = np.zeros([50000, num_classes])
		y_file = args.dataset+'/'+ args.arch +'/'+args.exp + "/y.npy"
		if os.path.isfile(y_file):
			y_all = np.load(y_file)
		else:
			y_all = []

	loadertrain = tqdm(trainloader, desc='{} E{:03d}'.format('train', epoch), ncols=0)
	epoch_loss = 0.0
	total=0.0
	clean_acc = 0.0
	adv_acc = 0.0
	for x_train, y_train, idx in loadertrain:
		n.eval()
		x_train, y_train = x_train.cuda(), y_train.cuda()

		y_pre = n(x_train)
		if args.method == 'pgd':
			logits_adv, loss = loss_functions.AT(n, x_train, y_train, optimizer, args)
		elif args.method == 'trades':
			logits_adv, loss = loss_functions.trades(n, x_train, y_train, optimizer, args)
		elif args.method == 'pgd_te':
			logits_adv, loss = pgd_te(x_train, y_train, idx, epoch+1, n, optimizer, weight)
		elif args.method == 'trades_te':
			logits_adv, loss = trades_te(x_train, y_train, idx, epoch+1, n, optimizer, weight)
		elif args.method == 'pgd_sat':
			logits_adv, loss = loss_functions.SAT(n, x_train, y_train, idx, optimizer, criterion, epoch, args)
		elif args.method == 'trades_sat':
			logits_adv, loss = loss_functions.SAT_TRADES(n, x_train, y_train, idx, optimizer, criterion, epoch, args)
		elif args.method == 'pencil':
			logits_adv, loss, new_y, yy = loss_functions.PENCIL(n, x_train, y_train, idx, new_y, y_all, optimizer, epoch, args)
		elif args.method == 'labelcorr':
			logits_adv, loss = loss_functions.LabelCorrection(n, x_train, y_train, optimizer, epoch+1, guidedMixup_ep, bootstrap_ep_mixup, k, temp_length, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args)
		elif args.method == 'elr':
			logits_adv, loss = loss_functions.ELR(n, x_train, y_train, idx, epoch, lambda_elr, criterion, optimizer, args)
		elif args.method == 'selfie':
			if epoch < warm_up:
				softmax_matrix = torch.softmax(y_pre.detach(), dim=1)
				correcter.async_update_prediction_matrix(idx, softmax_matrix)
				logits_adv, loss = loss_functions.SELFIE(n, x_train, y_train, optimizer, args)
			else:
				correcter.patch_clean_with_refurbishable_sample_batch(idx)
				y_train = torch.Tensor(correcter.train_set.targets[idx]).long().cuda()
				logits_adv, loss = loss_functions.SELFIE(n, x_train, y_train, optimizer, args)
				softmax_matrix = torch.softmax(y_pre.detach(), dim=1)
				correcter.async_update_prediction_matrix(idx, softmax_matrix)
		elif args.method == 'plc':
			if epoch <= warm_up:
				logits_adv, loss = loss_functions.PLC(n, x_train, y_train, optimizer, args)
				f_record[epoch % rollwindow, idx] = F.softmax(y_pre.detach().cpu(), dim=1)
			else:
				y_train = y_corrected[idx].long().cuda()
				logits_adv, loss = loss_functions.PLC(n, x_train, y_train, optimizer, args)
				f_record[epoch % rollwindow, idx] = F.softmax(y_pre.detach().cpu(), dim=1)
		loss.backward()
		optimizer.step()
		if args.method == 'pencil':
			if epoch >= args.stage1 and epoch < args.stage2:
				lambda1 = args.lambda1
				# update y_tilde by back-propagation
				yy.data.sub_(lambda1 * yy.grad.data)
				new_y[idx, :] = yy.data.cpu().numpy()
		epoch_loss += loss.data.item()
		_, predicted = torch.max(y_pre.data, 1)
		_, predictedadv = torch.max(logits_adv.data, 1)
		total += y_train.size(0)
		clean_acc += predicted.eq(y_train.data).cuda().sum()
		adv_acc += predictedadv.eq(y_train.data).cuda().sum()
		fmt = '{:.4f}'.format
		loadertrain.set_postfix(loss=fmt(loss.data.item()),
								acc_cl=fmt(clean_acc.item() / total * 100),
								acc_adv=fmt(adv_acc.item() / total * 100))
	train_clean_acc.append(clean_acc.item() / total * 100)
	train_adv_acc.append(adv_acc.item() / total * 100)
	scheduler.step()
	if args.method == 'pencil':
		if epoch < args.stage2:
			# save y_tilde
			y = new_y
			y_file = args.dataset + '/' + args.arch + '/' + args.exp + "/y.npy"
			np.save(y_file, y)
	if args.method == 'labelcorr':
		bmm_model, bmm_model_maxLoss, bmm_model_minLoss = loss_functions.track_training_loss(n, loadertrain)
		k = k + 1
		k = min(k, temp_length - 1)
	if args.method == 'plc':
		if epoch >= warm_up:
			f_x = f_record.mean(0)
			y_tilde = trainset.targets
			y_corrected, current_delta = loss_functions.lrt_correction(np.array(y_tilde).copy(), f_x, current_delta=current_delta,
														delta_increment=inc)

	if (epoch) % 1 == 0:
		Loss_test = nn.CrossEntropyLoss().cuda()
		test_loss_cl = 0.0
		test_loss_adv = 0.0
		correct_cl = 0.0
		correct_adv = 0.0
		total = 0.0
		n.eval()
		pgd_eval = torchattacks.PGD(n, eps=8.0/255.0, steps=20)
		loadertest = tqdm(testloader, desc='{} E{:03d}'.format('test', epoch), ncols=0)
		with torch.enable_grad():
			for x_test, y_test in loadertest:
				x_test, y_test = x_test.cuda(), y_test.cuda()
				x_adv = pgd_eval(x_test, y_test)
				n.eval()
				y_pre = n(x_test)
				y_adv = n(x_adv)
				loss_cl = Loss_test(y_pre, y_test)
				loss_adv = Loss_test(y_adv, y_test)
				test_loss_cl += loss_cl.data.item()
				test_loss_adv += loss_adv.data.item()
				_, predicted = torch.max(y_pre.data, 1)
				_, predicted_adv = torch.max(y_adv.data, 1)
				total += y_test.size(0)
				correct_cl += predicted.eq(y_test.data).cuda().sum()
				correct_adv += predicted_adv.eq(y_test.data).cuda().sum()
				fmt = '{:.4f}'.format
				loadertest.set_postfix(loss_cl=fmt(loss_cl.data.item()),
									   loss_adv=fmt(loss_adv.data.item()),
									   acc_cl=fmt(correct_cl.item() / total * 100),
									   acc_adv=fmt(correct_adv.item() / total * 100))
			test_clean_acc.append(correct_cl.item() / total * 100)
			test_adv_acc.append(correct_adv.item() / total * 100)
		if correct_adv.item() / total * 100 > best_eval_acc:
			best_eval_acc = correct_adv.item() / total * 100
			checkpoint = {
					'state_dict': n.state_dict(),
					'epoch': epoch
				}
			torch.save(checkpoint, args.save+ 'best.pkl')
checkpoint = {
			'state_dict': n.state_dict(),
			'epoch': epoch
		}
torch.save(checkpoint, args.save + 'last.pkl')
np.save(args.save+'_train_acc_cl.npy', train_clean_acc)
np.save(args.save+'_train_acc_adv.npy', train_adv_acc)
np.save(args.save+'_test_acc_cl.npy', test_clean_acc)
np.save(args.save+'_test_acc_adv.npy', test_adv_acc)