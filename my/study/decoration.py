class IndexMixin:
  # city = '南京'

  # def __init__(self,addr, email):
  #   self.addr = addr
  #   self.email = email
  #   print('Base Class')

  def __getitem__(self, key):
    return self.__dict__.get(key)

  def __setitem__(self, key, value):
    self.__dict__[key] = value

class Student():
  nation = '中国'
  def __init__(self,name,age):
    self.name = name
    self.age = age
    # super(Student, self).__init__('苜蓿园大街', 'koa.guo@gmail.com')
  @property
  def name(self):
    return self.__name
  
  @name.setter
  def name(self, value):
    self.__name = value
  
stu = Student('Allen', 24)


# stu['name'] = 'Koda'
# stu['age'] = '40'
stu.name = 'Alice'
# stu.set_name('Judy')

print(f"{stu.name}")

print(stu.__dict__)
# print(Student.nation)
# print(Student.city)
