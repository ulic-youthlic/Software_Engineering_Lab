class Dog:
    #类属性（所有实例共享）
    species="Canis"
    #构造方法（初始化实例属性）
    def __init__(self,name,age):
        self.name=name#实例属性
        self.age=age


#实例化对象
my_dog=Dog("buddy",3)
print(my_dog.name)

#类属性访问及控制
#类属性：定义在类内部但不在方法中，所有实例共享
#实例属性：通过self.属性名在__init__中定义，每个实例独有
#动态属性：允许动态为实例添加属性（比如my_dog.color="brown"）,但是可以通过__slots__限制允许的属性名（比如__slots__=('brand','category'),只允许这两个属性）

#封装
#python没有严格的私有属性，但是约定用下划线表示私有性
#_protected_var:暗示它受保护，__private_var暗示它私有

#类的方法：
#实例方法：第一个参数是self,表示实例本身，用于操作实例属性
'''
    def bark(self):
        return f"{self.name} says woof"
'''
#类方法：用@classmethod装饰器，参数是cls,可以操作类属性
'''
    @classmethod
    def get_species(cls)
    return cls.species
'''
#静态方法：用staticmethod装饰器，与类或实例无关
'''
    @staticmethod
    def is_puppy(age):
        return age<2
'''
#继承和多态:子类继承父类的属性和方法，并可以重写或拓展
'''
class Animak:
    def speak(self):
        print("Animal makes a sound")

class Cat(Animal):
    def speak(self): 重写父类方法
        print("Meow")
'''
#多态，不同子类对象通过相同方法名表现不同行为
'''
def make_sound(animal):
    animal.speak()

make_sound(Cat())
make_sound(Dog())
'''
#super()函数：用于调用父类的方法，常用于子类构造法中
'''
class Student(Person):
    def __init__(self,name,age,student_id):
        super().__init__(name,age)#调用父类构造方法
        self.student_id=student_id
'''
#特殊方法：
#__init__:对象初始化
#__str__:定义对象的字符串表示
#__add__运算符重载