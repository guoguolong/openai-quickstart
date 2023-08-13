from functools import wraps

class logger(object):  
    def __init__(self, level="INFO"):  
        self.level = level  
  
    def __call__(self, func):  
        @wraps(func)
        def wrapper(*args, **kwargs): 
            print("[{level}]: the function {func}() is running...".format(level=self.level, func=func.__name__)) 
            func(*args, **kwargs)  
  
        return wrapper  
  
  
@logger(level="ERROR")  
def say(something):  
    print("say {}!".format(something))  
  
  
say("hello2")
print(say.__name__)