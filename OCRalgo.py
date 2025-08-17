############# SOFTWARE PREPARATION #############


# installing dependencies
# pip install easyocr
# pip install git+https://github.com/JaidedAI/EasyOCR.git

# importing libraries
import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from PIL import Image
# from google.colab.patches import cv2_imshow

# this needs to run only once to load the model into memory
reader = easyocr.Reader(['ch_sim','en'])

# Get the list of all files
path = "/Users/omarkhawaldeh/Desktop/USF-JP project/dataset/OCR/fraud-nonfraud" #####RELATIVE PATH
originalChecksList = os.listdir(path)
originalChecksList = sorted(originalChecksList)
print("Number of checks in the dataset: ", len(originalChecksList))
# print(originalChecksList)


############# EXCTRACTING ROI #############

count = 0
for originalImg in originalChecksList:

  # making sure we're checking the right files
  if originalImg.__contains__(".jpg"):
    # image path
    checkImgPath = "dataset/OCR/fraud-nonfraud/" + originalImg #####RELATIVE PATH
    img = cv2.imread(checkImgPath)

    # resizing image
    img = cv2.resize(img, (900, 450))

    # cropping and displaying image
    if int(originalImg[5]) == 2:
      cropped_image = img[110:190, 680:875] # for type 2

    elif int(originalImg[5]) == 3:
      cropped_image = img[140:210, 680:880] # for type 3

    elif int(originalImg[5]) == 4:
      cropped_image = img[122:220, 720:875] # for type 4

    else:
      count+=1
      print(originalImg)
      continue

    # display cropped image
    # cv2.imshow(cropped_image)

    #saving image
    amountImage = originalImg[:6] + "-amount" + originalImg[6:]
    amountImagePath = "dataset/OCR/amount box/" + amountImage #####RELATIVE PATH
    cv2.imwrite(amountImagePath, cropped_image)
# print(count)

#dislaying the amount box file content
path = "dataset/OCR/amount box" #####RELATIVE PATH
amountChecksList = os.listdir(path)
amountChecksList = sorted(amountChecksList)
print("Number of amount checks in the dataset: " ,len(amountChecksList))
# print(amountChecksList)


# creating dictionary
ocrDict = {}
resultsDict = {}
def add(self, key, value):
  self[key] = value

############# OCR SCANNING #############

for amountImg in amountChecksList:

  # ignoring python generated files
  if amountImg.__contains__(".jpg"):

    #image path
    amountImgagePath = "dataset/OCR/amount box/" + amountImg #####RELATIVE PATH
    img = cv2.imread(amountImgagePath)

    # editing image
    # img = cv2.resize(img, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_CUBIC)
    # img = cv2.convertScaleAbs(img, alpha=1.1, beta=0.9)

    # ocr
    results = reader.readtext(img, allowlist ='0123456789', detail = 0)
    results = ''.join(results)

    # fixing up the scanned result
    # if results[len(results) - 2:] == "00":
    results = results[:len(results) - 2]

    # convert str to int
    checkNumber = int(amountImg[:4])

    # adding result to dict
    add(ocrDict, checkNumber, results)

    # # display image and result
    # cv2.imshow(img)
    # print("scanned integer:", results)

# printing dict
print("Number of amounts record: ", len(ocrDict))
# print(ocrDict)

# ############# SPEADSHEET CHECKER #############

# COMPARISON
# opening the CSV file
with open('dataset/OCR/Check Amounts.csv', mode ='r') as file: #####RELATIVE PATH

  # reading the CSV file
  csvFile = csv.reader(file)

  # looping through the contents of the CSV file
  for line in csvFile:

    # converting str to int
    try:
       line[0] = int(line[0])
    except ValueError:
       continue

    # check if we have a seriel number for this record
    if line[0] in ocrDict.keys():

      # modifying str
      line[1] = line[1][:len(line[1])-3]

      # check if results match records and add answer to results dict
      if line[1] == ocrDict[line[0]]:

        # print(line[0], line[1], ocrDict[line[0]], "perfect match")
        resultsDict[line[0]] = "PC" # perfect match

      else:

        # print(line[0], line[1], ocrDict[line[0]], "manual checking")
        resultsDict[line[0]] = "MC" # manual checking

      # print("-----------------------")

    else:
      continue

# print(ocrDict)
# print(resultsDict)

#TUNER
tuner = 10000

with open('dataset/OCR/Check Amounts.csv', mode ='r') as file: #####RELATIVE PATH
# reading the CSV file
  csvFile = csv.reader(file)

  # looping through the contents of the CSV file
  for line in csvFile:

    # modifying str
    line[1] = line[1][:len(line[1])-3]

    # converting str to int
    try:
       line[0] = int(line[0])
       line[1] = int(line[1])
    except ValueError:
       continue

    # check if we have a seriel number for this record
    if line[0] in ocrDict.keys():

      if int(line[1]) >= tuner:
        resultsDict[line[0]] = "MC" # manual checking
        # print(line[0], line[1], ocrDict[line[0]], "manual checking")

      else:
        continue

# print(ocrDict)
# print(resultsDict)


#CATEGORIZING
counter = 0
PC = 0
MC = 0

# looping through the result dict
for key in resultsDict.keys():
  if (resultsDict[key] != ""):

    counter += 1

    if (resultsDict[key] == "PC"):
      PC += 1
    else:
      MC += 1

# printing the results
print("Total:", counter, "| PC:", PC, "| MC:", MC)
print("Percent PC: %", (100*(PC/counter)))
print("Percent MC: %", (100*(MC/counter)))

#CONFUSION MATRIX
counter = 0
TP, FP, TN, FN = 0, 0, 0, 0
count = 0
with open('dataset/OCR/Check Amounts.csv', mode ='r') as file: #####RELATIVE PATH

  # reading the CSV file
  csvFile = csv.reader(file)

  # displaying the contents of the CSV file
  for line in csvFile:
    try:
       line[0] = int(line[0])
    except ValueError:
       continue

    if line[0] in resultsDict.keys() and resultsDict[line[0]] != "":
      # print(line[0], line[3], resultsDict[line[0]])
      counter += 1
      if (line[3] == "TRUE"):
        count+=1
      # comparing result to record and creating confusion matrix
      if (line[3] == "FALSE") and (resultsDict[line[0]] == "PC"):
        TN += 1
      elif (line[3] == "TRUE") and (resultsDict[line[0]] == "MC" or resultsDict[line[0]] == "SNDE"):
        TP += 1
      elif (line[3] == "FALSE") and (resultsDict[line[0]] == "MC" or resultsDict[line[0]] == "SNDE"):
        FP += 1
      elif (line[3] == "TRUE") and (resultsDict[line[0]] == "PC"):
        FN += 1
        # print(line)

print("count", counter, "| TP", TP, "| TN", TN, "| FP", FP, "| FN", FN)
print("Percent TP: %", (100*(TP/counter)))
print("Percent TN: %", (100*(TN/counter)))
print("Percent FP: %", (100*(FP/counter)))
print("Percent FN: %", (100*(FN/counter)))
# print(count)












###########SINGLE CHECK CHECKER##############
# # choose image path
# originalImg = "0108-2.jpg"
# imagePath = "dataset/OCR/fraud-nonfraud/" + originalImg #####RELATIVE PATH

# img = cv2.imread(imagePath)
# img = cv2.resize(img, (900, 450))

# # print("Original check")
# # cv2.imshow('image', img)

# ##### amount numuber extraction #####

# # cropping and displaying image
# if int(originalImg[5]) == 2:
#   cropped_image = img[110:190, 680:875] # for type 2

# elif int(originalImg[5]) == 3:
#   cropped_image = img[140:210, 680:880] # for type 3

# elif int(originalImg[5]) == 4:
#   cropped_image = img[122:220, 720:875] # for type 4


# # OCR scanning
# resultAmount = reader.readtext(cropped_image, allowlist ='0123456789', detail = 0)
# resultAmount = ''.join(resultAmount)
# # if resultAmount[len(resultAmount) - 2:] == "00":
# resultAmount = resultAmount[:len(resultAmount) - 2]

# # display cropped image
# print("\nAmount number: scanned integer->", resultAmount)
# # cv2.imshow('image', cropped_image)



# # opening the CSV file
# with open('dataset/OCR/Check Amounts.csv', mode ='r') as file: #####RELATIVE PATH

#   # reading the CSV file
#   csvFile = csv.reader(file)

#   # displaying the contents of the CSV file
#   for line in csvFile:

#     # convert str to int
#     try:
#        line[0] = int(line[0])
#     except ValueError:
#        continue

#     checkNumber = int(originalImg[:4])

#     # finding the specific transaction
#     if line[0] == checkNumber:

#       line[1] = line[1][:len(line[1])-3]

#       # do the scanned amount match the record?
#       if(line[1] == resultAmount):
#         print("\nThe amounts match!!! Here is our record for this transaction:")
#         print( line)

#       elif (line[1] != resultAmount):
#         print("\nThe amounts don't match, please send to manual checking. Here is our record for this transaction:")
#         print(line)