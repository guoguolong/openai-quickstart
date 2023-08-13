from abc import *

class MyClass(ABC):
	@abstractmethod
	def read(self):
		"""just comments"""


m = MyClass()
print(m.read())