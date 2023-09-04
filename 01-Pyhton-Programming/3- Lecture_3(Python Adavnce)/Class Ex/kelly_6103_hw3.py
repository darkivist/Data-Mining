#E.1

#Write a script to find duplicates from an array (define an array with some duplicates on it). If you use built in function in python explain the methods and how this methods are working.

#import libraries
import numpy as np
#create array
a = np.array([2,3,4,5,6,7,2,3,4,8,9,10])
#print array
print(a)
#use numpy unique function to find and print only unique members of the array (explain how this works later)
print(np.unique(a))

#E.2

#Write a script that finds all such numbers which are divisible by 2 and 5, less than 1000. If you use built in function in python explain the methods and how this methods are working.
#import libraries
import numpy as np

#create source array of numbers 1 through 5000
source_array = np.arange(5000)
#create empty destination array
results_array = []

#create conditional for selecting and appending i to destination array
for i in source_array:
    if i%2==0 and i%5==0 and i<1000:
        results_array.append(i)
#print completed destination array
print(results_array)

#E.3

#Write a Python class to convert a roman numeral to an integer. Hint: (Use the following symbols and numerals Wiki)

#Cite: https://www.w3resource.com/python-exercises/class-exercises/python-class-exercise-2.php

class py_solution:
    def roman_to_int(self, s):
        rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        int_val = 0
        for i in range(len(s)):
            if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
                int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
            else:
                int_val += rom_val[s[i]]
        return int_val

print(py_solution().roman_to_int('MMMCMLXXXVI'))
print(py_solution().roman_to_int('MMMM'))
print(py_solution().roman_to_int('C'))


#E.4

#Write a Python class to find sum the three elements of the given array to zero.

#Given: [-20, -10, -6, -4, 3, 4, 7, 10]
#Output : [[-10, 3, 7], [-6, -4, 10]]

#Cite: https://www.w3resource.com/python-exercises/class-exercises/python-class-exercise-6.php

class py_solution:
 def threeSum(self, nums):
        nums, result, i = sorted(nums), [], 0
        while i < len(nums) - 2:
            j, k = i + 1, len(nums) - 1
            while j < k:
                if nums[i] + nums[j] + nums[k] < 0:
                    j += 1
                elif nums[i] + nums[j] + nums[k] > 0:
                    k -= 1
                else:
                    result.append([nums[i], nums[j], nums[k]])
                    j, k = j + 1, k - 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
            i += 1
            while i < len(nums) - 2 and nums[i] == nums[i - 1]:
                i += 1
        return result

print(py_solution().threeSum([-20, -10, -6, -4, 3, 4, 7, 10]))