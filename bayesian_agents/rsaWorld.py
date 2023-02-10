class RSA_World:

	def __init__(
		self,
		target,
		speaker,
		rationality="DEFAULTBAD",
		):

		self.target=target
		self.rationality=rationality
		self.speaker=speaker

	def __hash__(self):
		return hash((self.target,self.speaker,self.rationality))

	def __eq__(self,other):
		return self.target==other.target and self.speaker==other.speaker and self.rationality==other.rationality

	def set_values(self,values):

		self.target=values[0]
		self.rationality=values[1]

	def __repr__(self):
		return "<World image:%s rationality:%s speaker:%s>" % (self.target,self.rationality, self.speaker)
