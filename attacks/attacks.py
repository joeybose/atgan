import torch
from torch.autograd import Variable

class FGSM(object):
	def __init__(self, epsilon):
		self.epsilon = epsilon

	def attack(self, inputs):
		"""
		Given a set of inputs and epsilon, return the perturbed inputs (wrapped in Variable objects).
		"""
		adv_inputs = inputs.data + self.epsilon * torch.sign(inputs.grad.data)
		adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
		return Variable(adv_inputs, requires_grad=False)


