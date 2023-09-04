# =================================================================
# Class_Ex1:
# Write python program that converts seconds to
# (x Hour, x min, x seconds)
# ----------------------------------------------------------------

print('Please enter time in seconds:')
x = float(input())
print(type(x))
print ("In minutes:", x//60)
print ("In hours:", x//60//60)

# =================================================================
# Class_Ex2:
# Write a python program to print all the different arrangements of the
# letters A, B, and C. Each string printed is a permutation of ABC.
# ----------------------------------------------------------------

def permute(prefix, suffix):
    suffix_size = len(suffix)
    if suffix_size == 0:
        print(prefix)
    else:
        for i in range(0, suffix_size):
            new_pre = prefix + [suffix[i]]
            new_suff = suffix[:i] + suffix[i + 1:]
            permute(new_pre, new_suff)
def print_permutations(lst):
    permute([], lst)
def main():
    a = ['A','B','C']
    print_permutations(a)
main()
print('#',50*"-")

# =================================================================
# Class_Ex3:
# Write a python program to print all the different arrangements of the
# letters A, B, C and D. Each string printed is a permutation of ABCD.
# ----------------------------------------------------------------

def permute(prefix, suffix):
    suffix_size = len(suffix)
    if suffix_size == 0:
        print(prefix)
    else:
        for i in range(0, suffix_size):
            new_pre = prefix + [suffix[i]]
            new_suff = suffix[:i] + suffix[i + 1:]
            permute(new_pre, new_suff)
def print_permutations(lst):
    permute([], lst)
def main():
    a = ['A','B','C','D']
    print_permutations(a)
main()
print('#',50*"-")

# =================================================================
# Class_Ex4:
# Suppose we wish to draw a triangular tree, and its height is provided
# by the user.
# ----------------------------------------------------------------

# Cite: https://stackoverflow.com/questions/43109056/drawing-a-right-triangle-python-3

tri_c = input('Enter character: ')
tri_h = int(input('Enter triangle height: '))
print('')
for j in range (tri_h):
    print((tri_c) * (j + 1))

# =================================================================
# Class_Ex5:
# Write python program to print prime numbers up to a specified values.
# ----------------------------------------------------------------

# Change this answer to reflect 4-Time_Example_2.py
# Cite https://studyfied.com/program/python-basic/prime-numbers-between-1-to-n/
# Input integer

print("Enter integer.")
x = int(input("Print prime numbers up to : "))
print("\nAll prime numbers between 2 and", x, "are : ")
for number in range(2, x + 1):
    i = 2
    for i in range(2, number):
        if(number % i == 0):
            i = number
            break;
    # If the number is prime then print it.
    if(i != number):
        print(number, end=" ")

# =================================================================