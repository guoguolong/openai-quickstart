class Num:
	_value = 20

	def __init__(self, v):
		# set_value(v)
		self._value = v

	@property
	def code(self):
		return self._value

	@code.setter
	def code(self, v):
		self._value = v
		
	def __str__(self) :
		return f"Num<{self._value}>"

	def __ge__(self, _a):
		return self._value >=  _a._value 

	def __gt__(self, _a):
		return self._value >  _a._value 

	def __add__(self, _a):
		return _a._value + self._value

	def __sub__(self, _a):
		return _a._value - self._value


n1 = Num(10)
n2 = Num(10)

print(n1 + n2)
print(n1 - n2)
print(n1 > n2)
# print(dir(n1))