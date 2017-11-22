import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys


def reduce_sum(x, keepdim=True):
	for a in reversed(range(1, x.dim())):
		x = x.sum(a, keepdim=keepdim)

	return x


def L2_dist(x, y):
	return reduce_sum((x - y)**2)


def torch_arctanh(x, eps=1e-6):
	x = x * (1. - eps)
	return (torch.log((1 + x) / (1 - x))) * 0.5


class FGSM(object):
	def __init__(self, epsilon=0.25):
		self.epsilon = epsilon

	def attack(self, inputs, labels, model, *args):
		"""
		Given a set of inputs and epsilon, return the perturbed inputs (as Variable objects),
		the predictions for the inputs from the model, and the percentage of inputs
		unsucessfully perturbed (i.e., model accuracy).

		The adversarial inputs is a python list of tensors.
		The predictions is a numpy array of classes, with length equal to the number of inputs.
		"""
		adv_inputs = self.perturb(inputs)
		predictions = torch.max(model(adv_inputs).data, 1)[1].cpu().numpy()
		num_unperturbed = (predictions == labels.data.cpu().numpy()).sum()
		return adv_inputs, predictions, num_unperturbed

	def perturb(self, inputs):
		"""
		Given a set of inputs and epsilon, return the perturbed inputs (as Variable objects).
		"""
		adv_inputs = inputs.data + self.epsilon * torch.sign(inputs.grad.data)
                adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
                adv_inputs = Variable(adv_inputs, requires_grad=False)
		return adv_inputs


class CarliniWagner(object):
	def __init__(self, confidence=0, learning_rate=1e-3, binary_search_steps=5, max_iterations=1000,
		initial_const=1, num_labels=10, clip_min=-1, clip_max=1, verbose=False):
		"""
		Return a tensor that constructs adversarial examples for the given input.
		Only supports untargeted attacks.

		- confidence : Confidence of adversarial examples: higher produces examples
			with larger l2 distortion, but more strongly classified as adversarial.
			Set to 0 in the paper.
		- learning_rate : The learning rate for the attack algorithm. Smaller values produce
			better results but are slower to converge.
		- binary_search_steps : The number of times we perform binary search to find the
			optimal tradeoff-constant between norm of the perturbation and confidence
			of the classification.
		- max_iterations : The maximum number of iterations. Setting this to a larger value
			will produce lower distortion results. Using only a few iterations requires
			a larger learning rate, and will produce larger distortion results.
		- initial_const : The initial tradeoff-constant to use to tune the relative
			importance of size of the perturbation and confidence of classification.
			If binary_search_steps is large, the initial constant is not important.
			A smaller value of this constant gives lower distortion results.
			This is c in the formula in the paper.
		- clip_min : Minimum input component value.
		- clip_max : Maximum input component value.
		- num_labels : Number of classes in the model's output.
		- verbose : Print output in detail.
		"""
		self.confidence = confidence
		self.learning_rate = learning_rate
		self.initial_const = initial_const
		self.num_labels = num_labels

		self.binary_search_steps = binary_search_steps
		self.repeat = binary_search_steps >= 10
		self.max_iterations = max_iterations

		# allows early aborts if gradient descent is unable to make progress
		self.abort_early = True
		self.verbose = verbose

		self.clip_min = clip_min
		self.clip_max = clip_max
		self.cuda = torch.cuda.is_available()

	def _compare(self, prediction, label):
		"""
		Return True if label is not the most likely class.
		If there is a prediction for each class, prediction[label] should be at least
		self.confidence from being the most likely class.
		"""
		if not isinstance(prediction, (float, int, np.int64)):
			prediction = np.copy(prediction)
			prediction[label] += self.confidence
			prediction = np.argmax(prediction)

		return prediction != label

	def _optimize(self, model, optimizer, modifier, inputs, labels, scale_const):
		"""
		Calculate loss and optimize for modifier here. Return the loss, adversarial inputs,
		and predicted classes. Since the attack is untargeted, aim to make label the least
		likely class.

		modifier is the variable we're optimizing over (w in the paper).
		Don't think of it as weights in a NN; there is a unique w for each x in the batch.
		"""
		inputs_adv = (torch.tanh(modifier + inputs) + 1) * 0.5
		inputs_adv = inputs_adv * (self.clip_max - self.clip_min) + self.clip_min
		# outputs BEFORE SOFTMAX
		predicted = model(inputs_adv)

		# before taking the L2 distance between the original and perturbed inputs,
		# transform the original inputs in the same way (arctan, then clip)
		unmodified = (torch.tanh(inputs) + 1) * 0.5
		unmodified = unmodified * (self.clip_max - self.clip_min) + self.clip_min
		dist = L2_dist(inputs_adv, unmodified)
		loss2 = dist.sum()

		# compute probability of label class and maximum other
		real = (labels * predicted).sum(1)
		other = ((1. - labels) * predicted - labels * 10000.).max(1)[0]

		# the greater the likelihood of label, the greater the loss
		loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
		loss1 = torch.sum(scale_const * loss1)
		loss = loss1 + loss2

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# convert to numpy form before returning it
		loss = loss.data.cpu().numpy()[0]
		dist = dist.data.cpu().numpy()
		predicted = predicted.data.cpu().numpy()
		# inputs_adv = inputs_adv.data.permute(0, 2, 3, 1).cpu().numpy()

		return loss, dist, predicted, inputs_adv

	def attack(self, inputs, labels, model, *args):
		"""
		Given a set of inputs, labels, and the model, return the perturbed inputs (as Variable objects).
		inputs and labels should be Variable objects themselves.
		"""
		inputs = inputs.clone()
		labels = labels.clone()

		batch_size = inputs.size(0)
		labels = labels.data

		# re-scale instances to be within range [0, 1]
		input_vars = (inputs.data - self.clip_min) / (self.clip_max - self.clip_min)
		input_vars = torch.clamp(input_vars, 0., 1.)
		# now convert to [-1, 1]
		input_vars = (input_vars * 2) - 1
		# convert to tanh-space
		input_vars = input_vars * .999999
		input_vars = (torch.log((1 + input_vars) / (1 - input_vars))) * 0.5 # arctanh
		input_vars = Variable(input_vars, requires_grad=False)

		# set the lower and upper bounds accordingly
		lower_bound = np.zeros(batch_size)
		scale_const = np.ones(batch_size) * self.initial_const
		upper_bound = np.ones(batch_size) * 1e10

		# numpy placeholders for the overall best l2, most likely label, and adversarial image
		o_best_l2 = [1e10] * batch_size
		o_best_score = [-1] * batch_size
		o_best_attack = inputs.clone()

		# one-hot encoding of labels
		one_hot_labels = torch.zeros(labels.size() + (self.num_labels,))
		if self.cuda: one_hot_labels = one_hot_labels.cuda()
		one_hot_labels.scatter_(1, labels.unsqueeze(1), 1.)
		label_vars = Variable(one_hot_labels, requires_grad=False)

		# setup the modifier variable; this is the variable we are optimizing over
		modifier = torch.zeros(inputs.size()).float()
		modifier_var = Variable(modifier.cuda() if self.cuda else modifier, requires_grad=True)

		optimizer = optim.Adam([modifier_var], lr=self.learning_rate)

		for outer_step in range(self.binary_search_steps):
			if self.verbose: print '\nsearch step: {0}'.format(outer_step)
			best_l2 = [1e10] * batch_size
			best_score = [-1] * batch_size

			# last iteration (if we run many steps) repeat the search once
			if self.repeat and outer_step == self.binary_search_steps - 1:
				scale_const = upper_bound

			scale_const_tensor = torch.from_numpy(scale_const).float()	# .float() needed to conver to FloatTensor
			scale_const_var = Variable(scale_const_tensor.cuda() if self.cuda else scale_const_tensor, requires_grad=False)

			prev_loss = 1e3 	# for early abort

			for step in range(self.max_iterations):
				loss, dist, predicted, input_adv = self._optimize(model, optimizer, modifier_var,
					input_vars, label_vars, scale_const_var)

				if step % 10 == 0 or step == self.max_iterations - 1:
					if self.verbose: print "Step: {0:>4}, loss: {1:6.6f}, dist: {2:8.6f}, modifier mean: {3:.6e}".format(
						step, loss, dist.mean(), modifier_var.data.mean())


				# abort early if loss is too small
				if self.abort_early and step % (self.max_iterations // 10) == 0:
					if loss > prev_loss * 0.9999:
						if self.verbose: print 'Aborting early...'
						break

					prev_loss = loss

				# update best result for each image
				for i in range(batch_size):
					y_hat = np.argmax(predicted[i])
					y = labels[i]

					# if smaller perturbation and still different predicted class ...
					if dist[i] < best_l2[i] and self._compare(y_hat, y):
						best_l2[i] = dist[i]
						best_score[i] = y_hat

					# update overall best results
					if dist[i] < o_best_l2[i] and self._compare(y_hat, y):
						o_best_l2[i] = dist[i]
						o_best_score[i] = y_hat
						o_best_attack[i] = input_adv[i]

				sys.stdout.flush()

			# adjust constants
			batch_failure, batch_success = 0, 0

			for i in range(batch_size):
				if self._compare(best_score[i], labels[i]) and best_score[i] != -1:
					# successful, do binary search and divide const by two
					upper_bound[i] = min(upper_bound[i], scale_const[i])

					if upper_bound[i] < 1e9:
						scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
				else:
					# failure, multiply by 10 if no solution found
					# or do binary search with the known upper bound
					lower_bound[i] = max(lower_bound[i], scale_const[i])
					upper_bound[i] = (lower_bound[i] + upper_bound[i]) / 2 if (upper_bound[i] < 1e9) else (scale_const[i] * 10)

				if self._compare(o_best_score[i], labels[i]) and o_best_score[i] != -1:
					batch_success += 1
				else:
					batch_failure += 1

			if self.verbose: print 'failures: {0} successes: {1}'.format(batch_failure, batch_success)
			sys.stdout.flush()

		# if no good adv attack, then equivalent to using base image
		for i in range(len(o_best_score)):
			if o_best_score[i] == -1:
				o_best_score[i] = labels[i]

		o_best_score = np.array(o_best_score)
		num_unperturbed = (o_best_score == labels.cpu().numpy()).sum()

		return o_best_attack, o_best_score, num_unperturbed


class DCGAN(object):
	def __init__(self, num_channels=3, ngf=100, cg=0.2, learning_rate=1e-4, train_adv=False):
		"""
		Initialize a DCGAN. Perturbations from the GAN are added to the inputs to
		create adversarial attacks.

		- num_channels is the number of channels in the input
		- ngf is size of the conv layers
		- cg is the normalization constant for perturbation (higher means encourage smaller perturbation)
		- learning_rate is learning rate for generator optimizer
		- train_adv is whether the model being attacked should be trained adversarially
		"""
		self.generator = nn.Sequential(
			# input is (nc) x 32 x 32
			nn.Conv2d(num_channels, ngf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 1, 1, 0, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 3 x 32 x 32
			nn.Conv2d(ngf, num_channels, 1, 1, 0, bias=False),
			nn.Tanh()
		)

		self.cuda = torch.cuda.is_available()

		if self.cuda:
			self.generator.cuda()
			self.generator = torch.nn.DataParallel(self.generator, device_ids=range(torch.cuda.device_count()))
			cudnn.benchmark = True

		self.criterion = nn.CrossEntropyLoss()
		self.cg = cg
		self.optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)
		self.train_adv = train_adv

	def attack(self, inputs, labels, model, model_optimizer=None, epsilon=1.0, *args):
                """
                Given a set of inputs, return the perturbed inputs (as Variable objects),
                the predictions for the inputs from the model, and the percentage of inputs
                unsucessfully perturbed (i.e., model accuracy).

		If self.train_adversarial is True, train the model adversarially.

                The adversarial inputs is a python list of tensors.
                The predictions is a numpy array of classes, with length equal to the number of inputs.
                """
		perturbation = self.generator(Variable(inputs.data))
		adv_inputs = inputs + epsilon * perturbation
		adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)

		predictions = model(adv_inputs)
		loss = torch.exp(-1 * self.criterion(predictions, labels)) + \
                        self.cg * (torch.norm(perturbation, 2).data[0] ** 2) 

		# optimizer step for the generator
		self.optimizer.zero_grad()
		loss.backward(retain_graph=True)
		self.optimizer.step()

		# optimizer step for the discriminator (if training adversarially)
		if self.train_adv and model_optimizer:
			discriminator_loss = self.criterion(predictions, labels)
			model_optimizer.zero_grad()
			discriminator_loss.backward()
			model_optimizer.step()

		# print perturbation.data.mean(), inputs.data.mean()
		# print loss.data[0], torch.norm(perturbation, 2).data[0], torch.norm(inputs, 2).data[0]

		# prep the predictions and inputs to be returned
		predictions = torch.max(predictions.data, 1)[1].cpu().numpy()
		num_unperturbed = (predictions == labels.data.cpu().numpy()).sum()

		return adv_inputs, predictions, num_unperturbed

	def perturb(self, inputs, epsilon=1.0):
		perturbation = self.generator(Variable(inputs.data))
		adv_inputs = inputs + epsilon * perturbation
		adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)

		return adv_inputs

	def save(self, fn):
		torch.save(self.generator.state_dict(), fn)

	def load(self, fn):
		self.generator.load_state_dict(torch.load(fn))
