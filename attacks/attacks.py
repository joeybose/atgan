import torch
import numpy as np
from torch.autograd import Variable

def reduce_sum(x, keepdim=True):
	for a in reversed(range(1, x.dim())):
		x = x.sum(a, keepdim=keepdim)

	return x


def L2_dist(x, y):
	return reduce_sum((x - y)**2)


class FGSM(object):
	def __init__(self, epsilon=0.25):
		self.epsilon = epsilon

	def attack(self, inputs):
		"""
		Given a set of inputs and epsilon, return the perturbed inputs (as Variable objects).
		"""
		adv_inputs = inputs.data + self.epsilon * torch.sign(inputs.grad.data)
		adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
		return Variable(adv_inputs, requires_grad=False)


class CarliniWagner(object):
	def __init__(self, confidence=0, learning_rate=1e-3, binary_search_steps=5, max_iterations=1000, 
		initial_const=0.01, num_labels=10, clip_min=-1, clip_max=1):
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
		"""
		self.confidence = confidence
		self.learning_rate = learning_rate
		self.initial_const = initial_const
		self.num_labels = num_labels
		self.shape = shape

		self.binary_search_steps = binary_search_steps
		self.repeat = binary_search_steps >= 10
		self.max_iterations = max_iterations
		
		# allows early aborts if gradient descent is unable to make progress 
		self.abort_early = True

		self.clip_min = clip_min
		self.clip_max = clip_max
		self.cuda = torch.cuda.is_available()

	def _compare(self, prediction, label):
		"""
		Return True if the most likely class, after adjusting for confidence, is not the label. 
		"""
		if not isinstance(output, (float, int, np.int64)):
			prediction = np.copy(output)	
			prediction[label] += self.confidence
		
		return np.argmax(prediction) != label 

	def _optimize(self, model, optimizer, modifier, inputs, labels, scale_const):
		"""
		Calculate loss and optimize for modifier here. Return the loss. 
		Since the attack is untargeted, aim to make label the least likely class.

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
		dist = L2_dist(input_adv, unmodified).sum()
		loss2 = dist		

		real = (labels * predicted).sum(1)
		other = ((1. - label) * predicted - label * 10000.).max(1)[0]

		# the greater the likelihood of label, the greater the loss
		loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
		loss1 = torch.sum(scale_const * loss1)
		loss = loss1 + loss2 		

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		return loss.data[0] 

