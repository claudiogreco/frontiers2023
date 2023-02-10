import numpy as np
from utils.image_and_text_utils import max_sentence_length,vectorize_caption

class RSA_State:

	def __init__(
		self,
		initial_world_prior,
		listener_rationality=1.0
		):

		self.context_sentence=np.expand_dims(np.expand_dims(vectorize_caption("")[0],0),-1)
		self.dim = {"image":0,"rationality":1,"speaker":2}
		self.world_priors = np.asarray([initial_world_prior for x in range(max_sentence_length+1)])
		self.listener_rationality=listener_rationality
		self.timestep=1


	def __hash__(self):
		return hash((self.timestep,self.listener_rationality,tuple(self.context_sentence)))
