try:
	1 / 2
	print('No by zero')
except Exception as e:
	print('Error: ', e)
finally:
	print('Anyway ,you run')