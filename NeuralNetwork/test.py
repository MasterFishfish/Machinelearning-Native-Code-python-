import random

import sympy
import numpy

def pdtest():
    x = sympy.symbols("x")
    y = x ** 2
    dify = sympy.diff(y, x).subs({x: 3})
    print(type(dify))
    print(dify)


class Person(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.weight = 'weight'

    def talk(self):
        print("person is talking....")


class Chinese(Person):
    def __init__(self, name, age, language):
        Person.__init__(self, name, age)
        self.__language = language
        print(self.name, self.age, self.weight, self.__language)

    def talk(self):  # 子类 重构方法
        print('%s is speaking chinese' % self.name)

    def walk(self):
        print('is walking...')


if __name__ == "__main__":
    c = Chinese('bigberg', 22, 'Chinese')
    d = Chinese('zzz', 23, 'aaa')
    a = random.random()
    array = []
    for i in range(3):
        array.append(random.random())
    print(array)

    g = [1, 2, 3]
    t = [2, 3, 4]
    print(numpy.dot(g, t))

    outdata = 2
    target_data = 3
    x = sympy.symbols("x")
    y = 0.5 * (( target_data - x ) ** 2.0)
    error_out_dify = sympy.diff(y, x).subs({x: outdata})
    print(error_out_dify)