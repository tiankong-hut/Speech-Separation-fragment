# -*- coding:utf-8 -*-
# https://www.cnblogs.com/chengd/articles/7287528.html
# 面向对象的三大特性是指：封装、继承和多态

# ############ 定义实现功能的类 ###########
print("''''''类'''''''")

class Person:

    def __init__(self, na, gen, age, fig):
        self.name = na
        self.gender = gen
        self.age = age
        self.fight =fig

    def grassland(self):
    #注释：消耗200战斗力
        self.fight = self.fight - 200

    def practice(self):
    #注释：增长100战斗力
        self.fight = self.fight + 200

    def incest(self):
    #注释：消耗500战斗力"""
        self.fight = self.fight - 500

    def detail(self):
    #"""注释：当前对象的详细情况"""
        temp = "姓名:%s ; 性别:%s ; 年龄:%s ; 战斗力:%s" % (self.name, self.gender, self.age, self.fight)
        print (temp)


# ##################### 开始游戏 #####################

cang = Person('A', '女', 18, 1000) # 创建A角色
dong = Person('B', '男', 20, 1800) # 创建B角色
bo   = Person('C', '男', 19, 1500) # 创建C角色

print("’‘’第一次战斗‘’‘")
cang.incest()   #A参加一次多人游戏
dong.practice() #B自我修炼了一次
bo.grassland()  #C参加一次草丛战斗


#输出当前所有人的详细情况
# print(A.fig)
cang.detail()
dong.detail()
bo.detail()

print("’‘’新战斗‘’‘")
cang.incest() #A又参加一次
dong.incest() #B又参加一次
bo.practice() #C又参加一次

#输出当前所有人的详细情况
cang.detail()
dong.detail()
bo.detail()



'''''''继承'''''''
print("'''''''继承'''''''")
# 动物：吃、喝、拉、撒
# 猫：喵喵叫（猫继承动物的功能）
# 狗：汪汪叫（狗继承动物的功能）
class Animal:
    def eat(self):
        print ("%s 吃 " % self.name)

    def drink(self):
        print ("%s 喝 " % self.name)

    def shit(self):
        print ("%s 拉 " % self.name)

    def pee(self):
        print ("%s 撒 " % self.name)


class Cat(Animal):       # 在类后面括号中写入另外一个类名，表示当前类继承另外一个类
    def __init__(self, name):
        self.name = name
        self.breed = '猫'

    def cry(self):
        print
        '喵喵叫'


class Dog(Animal):       #当前类Dog继承另外一个类:Animal
    def __init__(self, name):
        self.name = name
        self.breed = '狗'

    def cry(self):
        print
        '汪汪叫'


# ######### 执行 #########

c1 = Cat('小白家的小黑猫:')
c1.eat()

c2 = Cat('小黑的小白猫: ')
c2.drink()

d1 = Dog('胖子家的小瘦狗:')
d1.pee()



