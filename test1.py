class Person():
    def __init__(self, name):
        self.myname = name
    def hello(self):
        print('Hello! I am {}.'.format(self.myname))
def main():
    Peter = Person('Peter')
    Mary = Person('Mary')
    Mary.hello()

if __name__ == '__main__' :
    main()