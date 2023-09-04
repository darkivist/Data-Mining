# =================================================================
# Class_Ex1:
# Writes a python script (use class) to simulate a Stopwatch .
# push a button to start the clock (call the start method), push a button
# to stop the clock (call the stop method), and then read the elapsed time
# (use the result of the elapsed method).
# ----------------------------------------------------------------

#Cite: Class notes

from time import time
print("Hit enter to turn on timer: ", end="")
activate = time()
name = input()
print("Hit enter to start timer: ", end="")
start_time = time()
name = input()
print("Hit enter to stop timer: ", end="")
stop_time = time()
name = input()
elapsed = stop_time - start_time
print(name, "Elapsed time is", elapsed, "seconds.")

# =================================================================
# Class_Ex2:
# Write a python script (use class)to implement pow(x, n).
# ----------------------------------------------------------------

#Cite: https://www.w3resource.com/python-exercises/class-exercises/python-class-exercise-7.php

class py_solution:
   def pow(self, x, n):
        if x==0 or x==1 or n==1:
            return x

        if x==-1:
            if n%2 ==0:
                return 1
            else:
                return -1
        if n==0:
            return 1
        if n<0:
            return 1/self.pow(x,-n)
        val = self.pow(x,n//2)
        if n%2 ==0:
            return val*val
        return val*val*x

print(py_solution().pow(2, -3));
print(py_solution().pow(3, 5));
print(py_solution().pow(100, 0));

# =================================================================
# Class_Ex3:
# Write a python class to calculate the area of rectangle by length
# and width and a method which will compute the area of a rectangle.
# ----------------------------------------------------------------

#Cite: https://www.w3resource.com/python-exercises/class-exercises/python-class-exercise-10.php

class Rectangle():
    def __init__(self, l, w):
        self.length = l
        self.width  = w

    def rectangle_area(self):
        return self.length*self.width

newRectangle = Rectangle(12, 10)
print(newRectangle.rectangle_area())

# =================================================================
# Class_Ex4:
# Write a python class and name it Circle to calculate the area of circle
# by a radius and two methods which will compute the area and the perimeter
# of a circle.
# ----------------------------------------------------------------

#Cite: https://www.w3resource.com/python-exercises/class-exercises/python-class-exercise-11.php

class Circle():
    def __init__(self, r):
        self.radius = r

    def area(self):
        return self.radius ** 2 * 3.14

    def perimeter(self):
        return 2 * self.radius * 3.14

NewCircle = Circle(8)
print(NewCircle.area())
print(NewCircle.perimeter())