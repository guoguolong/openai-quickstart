from functools import wraps

def log(func): 
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('LOG BEFRE')
        r = func(*args, **kwargs)
        print('LOG AFTER')
        return r
    return wrapper
    
# class A:
#     def echo(self, v):
#         print(v)

@log
def echo(v):
    print('Echo : ', v)

# echo('Hello')
# print(echo.__name__)

echo.__wrapped__('Yes')