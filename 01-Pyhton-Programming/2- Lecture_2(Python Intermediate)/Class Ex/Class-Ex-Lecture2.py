# =================================================================
# Class_Ex1:
# Write a program that simulates the rolling of a die.
# ----------------------------------------------------------------




# =================================================================
# Class_Ex2:
# Answer  Ex1 by using functions.
# ----------------------------------------------------------------

from random import randrange
print('Hit enter to roll die: ')
x = input()
print(randrange(1,6))
print('#',50*"-")

# =================================================================
# Class_Ex3:
# Randomly Permuting a List
# ----------------------------------------------------------------

from random import randrange

def permute(prefix, suffix):
    suffix_size = len(suffix)
    if suffix_size == 0:
        print(prefix)
    else:
        for i in range(0, suffix_size):
            print(randrange(0, suffix_size), end=' ')
def print_permutations(lst):
    permute([], lst)
def main():
    a = [1, 2, 3, 4, 5, 6, 7]
    print_permutations(a)
main()

print('#',50*"-")

# =================================================================
# Class_Ex4:
# Write a program to convert a tuple to a string.
# ----------------------------------------------------------------

my_tuple = ("this", "is", "a", "tuple", "with", "number", 100.75)
print (type(my_tuple))
tuple_to_string = str(my_tuple)
print (type(tuple_to_string))
print (tuple_to_string)

# =================================================================
# Class_Ex5:
# Write a program to get the 3th element of a tuple.
# ----------------------------------------------------------------

another_tuple = ("this", "is", "another", "tuple")
print (another_tuple[2])

# =================================================================
# Class_Ex6:
# Write a program to check if an element exists in a tuple or not.
# ----------------------------------------------------------------

yet_another_tuple = ("here","is","one","more","exciting","tuple")
print('Enter element to search tuple for: ')
x = input()
if x in yet_another_tuple:
    print('Element found!')
else:
    print('Element not found.')

# =================================================================
# Class_Ex7:
# Write a  program to check a list is empty or not.
# ----------------------------------------------------------------

#len counts the character in the list, and if there are more than zero returns "not empty"

my_list = [1,2,3,4,5]
if len(my_list) ==0:
    print ("empty")
else:
    print ("not empty")

# =================================================================
# Class_Ex8:
# Write a program to generate a 4*5*3 3D array that each element is O.
# ----------------------------------------------------------------

