# coding=utf-8
# https://www.cnblogs.com/itcomputer/articles/4578769.html

# import timeit
# import math
# import pprint
# # 为了输出美观，使用了pprint这个有意思的模块
#
# def myfun():
#     for i in range(100):
#         for j in range(2, 10):
#             math.pow(i, 1/j)
#
# n = 100
#
# t1 = timeit.timeit(stmt=myfun, number=n)
# pprint.pprint(t1)
# t2 = timeit.repeat(stmt=myfun, number=n, repeat=5)
# pprint.pprint(t2)
#
# print()
#
# timeitObj = timeit.Timer(stmt=myfun)
# t3 = timeitObj.timeit(number=n)
# pprint.pprint(t3)
# t4 = timeitObj.repeat(number=n, repeat=5)
# pprint.pprint(t4)



# http://m.blog.csdn.net/justheretobe/article/details/50708890
# import random
# import timeit
# from time import clock
#
#
# def get_random_number(num):
#     '''get random number, no repeated element; use random.sample() method'''
#     return random.sample(range(num), num)
#
#
# if __name__ == "__main__":
#     #use clock() method to calculate time
#     start = clock()
#     list_a = get_random_number(200)
#     finish = clock()
#     print(finish - start)
#     #check the length of list generated by function
#     print(len(list_a))
#     print(len(set(list_a)))
#
#     #use timeit.Timer() method
#     t1 = timeit.Timer('get_random_number(200)',
#                       setup="from __main__ import get_random_number")
#     #only excute once
#     print(t1.timeit(1))
#     #only repeat once, and only excute once
#     print(t1.repeat(1, 1))
#
#     #use timeit.Timer() and lambda to invoke function
#     print(timeit.Timer(lambda: get_random_number(200)).timeit(1))

# http://blog.csdn.net/huludan/article/details/43935873
# def test1():
#     n=0
#     for i in range(101):
#         n+=i
#     return n
# def test2():
#     return sum(range(101))
# def test3():
#     return sum(x for x in range(101))
# if __name__=='__main__':
#     # from timeit import Timer
#     import timeit
#     t1=timeit.Timer("test1()","from __main__ import test1")
#     t2=timeit.Timer("test2()","from __main__ import test2")
#     t3=timeit.Timer("test3()","from __main__ import test3")
#     print t1.timeit(10000)
#     print t2.timeit(10000)
#     print t3.timeit(10000)
#     print t1.repeat(3,10000)
#     print t2.repeat(3,10000)
#     print t3.repeat(3,10000)

# https://segmentfault.com/q/1010000005924858/a-1020000005925007
# def is_unique_char(string):
#     if len(string) > 256:
#         return True
#
#     record = 0L
#
#     for ch in string:
#         # print record
#         ch_val = ord(ch)
#
#         if (record & (1 << ch_val)) > 0:
#             return False
#
#         record |= (1 << ch_val)
#
#     return True
#
# import string
# s1 = string.ascii_letters + string.digits
#
# if __name__ == '__main__':
#     import timeit
#     import time
#     print is_unique_char(s1)
#     print timeit.timeit(stmt="is_unique_char(s1)",
#                         setup="from __main__ import is_unique_char, s1")
#     # 79.7438879013s
#     btime = time.time()
#     is_unique_char(s1)
#     etime = time.time()
#     print (etime - btime)