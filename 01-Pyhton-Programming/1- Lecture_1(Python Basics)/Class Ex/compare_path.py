from pathlib import Path
from pathlib import PurePath
import os

# fileList = open("fileList.txt", "r")
# data = fileList.read()
# fileList_reformatted = data.replace('\n', '').split(",")
# print(fileList_reformatted)

p = Path('/Users/paulkelly/Downloads/compare').rglob('*')
filePaths = [x for x in p if x.is_file()]
filePaths_string = [str(x) for x in filePaths]
# print(filePaths_string)

# differences1 = []
# for element in fileList_reformatted:
#    if element not in filePaths_string:
#        differences1.append(element)

# print("The following files from the provided list were not found:", differences1)

# differences2 = []
# for element in filePaths_string:
#    if element not in fileList_reformatted:
#        differences2.append(element)

# print("The following unexpected files were found:", differences2)

wrong_location = []
for element in filePaths:
    if element.parts[-1].split("_")[0] != element.parent.parts[-1].split("_")[0]:
        wrong_location.append(element)

print("Following files may be in the wrong location:", wrong_location)

# wrong_location = []
# for element in p:
#    if element.str(Path.name) != element.str(Path.parent):
#        wrong_location.append(element)

# print("Following files may be in the wrong location:", wrong_location)

#p1 = Path('/Users/paulkelly/Downloads/compare/afc20222011/afc20222012_123')
#if p1.parts[-1].split("_")[0] != p1.parent.parts[-1].split("_")[0]:
#    print("files may be in wrong location")
#last_part = p1.parts[-1].split("_")[0]
#print(last_part)