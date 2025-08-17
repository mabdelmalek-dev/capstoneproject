############# SOFTWARE PREPARATION #############

#install tensorflow
#pip install tensorflow opencv-python matplotlib

#import libraries
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import imghdr
import shutil
#from google.colab.patches import cv2_imshow

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#confirm use of GPU
tf.config.list_physical_devices('GPU')

############# TRAINING SET #############

path = "dataset/ML/uncropped data/fraud" ##### RELATIVE PATH
UncroppedFraudList = os.listdir(path)
UncroppedFraudList = sorted(UncroppedFraudList)
print("Number of UncroppedFraudList: ", len(UncroppedFraudList))
# print(UncroppedFraudList)
# print("\n")

path = "dataset/ML/uncropped data/non fraud" ##### RELATIVE PATH
UncroppedNonFraudList = os.listdir(path)
UncroppedNonFraudList = sorted(UncroppedNonFraudList)
print("Number of UncroppedNonFraudList: ", len(UncroppedNonFraudList))
# print(UncroppedNonFraudList)


#EXTRACTING ROI
count = 0

# fraud checks
for originalImg in UncroppedFraudList:

  # making sure we're checking the right files
  if originalImg.__contains__(".jpg"):

    # image path
    checkImgPath = "dataset/ML/uncropped data/fraud/" + originalImg ##### RELATIVE PATH
    img = cv2.imread(checkImgPath)

    # resizing image
    img = cv2.resize(img, (900, 450))

    # cropping and displaying image
    if int(originalImg[5]) == 1:
      cropped_image = img[110:200, 110:660] # for type 1

    elif int(originalImg[5]) == 2:
      cropped_image = img[80:200, 130:545] # for type 2

    elif int(originalImg[5]) == 3:
      cropped_image = img[115:220, 135:630] # for type 3

    elif int(originalImg[5]) == 4:
      cropped_image = img[100:210, 100:680] # for type 4

    else:
      count +=1
      continue

    cropped_image = cv2.resize(cropped_image, (500, 100))

    # display cropped image
    # cv2.imshow("", cropped_image)

    #saving image
    croppedFilePath = "dataset/ML/data/fraud/" + originalImg ##### RELATIVE PATH
    cv2.imwrite(croppedFilePath, cropped_image)

# non fraud checks
for originalImg in UncroppedNonFraudList:

  # making sure we're checking the right files
  if originalImg.__contains__(".jpg"):

    # image path
    checkImgPath = "dataset/ML/uncropped data/non fraud/" + originalImg ##### RELATIVE PATH
    img = cv2.imread(checkImgPath)

    # resizing image
    img = cv2.resize(img, (900, 450))

    # cropping and displaying image
    if int(originalImg[5]) == 1:
      cropped_image = img[110:200, 110:660] # for type 1

    elif int(originalImg[5]) == 2:
      cropped_image = img[80:200, 130:545] # for type 2

    elif int(originalImg[5]) == 3:
      cropped_image = img[115:220, 135:630] # for type 3

    elif int(originalImg[5]) == 4:
      cropped_image = img[100:210, 100:680] # for type 4

    else:
      count +=1
      continue

    cropped_image = cv2.resize(cropped_image, (500, 100))

    # display cropped image
    # cv2.imshow(cropped_image)

    #saving image
    croppedFilePath = "dataset/ML/data/non fraud/" + originalImg ##### RELATIVE PATH
    cv2.imwrite(croppedFilePath, cropped_image)

# print(count)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#IMPORTING CROPPED DATA
data_dir = 'dataset/ML/data' ##### RELATIVE PATH

checkpoints_path_test = 'dataset/ML/data/.ipynb_checkpoints'
if os.path.exists(checkpoints_path_test):
    shutil.rmtree(checkpoints_path_test)

path = data_dir + "/fraud" ##### RELATIVE PATH
nonFraudList = os.listdir(path)
nonFraudList = sorted(nonFraudList)
print("Number of cropped non fraud list",len(nonFraudList))
# print("nonFraudList: ", nonFraudList)

path = data_dir + "/non fraud" ##### RELATIVE PATH
FraudList = os.listdir(path)
FraudList = sorted(FraudList)
print("Number of cropped fraud list",len(FraudList))
# print("FraudList: ", FraudList)

# # img = cv2.imread(os.path.join(data_dir, 'non_fraudulent', '0001-4.jpg'))
# # cv2_imshow(img)

#CLASSIFYING
# print(data_dir)
# for image_class in os.listdir(data_dir):
#   for image in os.listdir(os.path.join(data_dir, image_class)):
#     image_path = os.path.join(data_dir, image_class, image)

#     try:
#         img = cv2.imread(image_path)
#         tip = imghdr.what(image_path)

#     except Exception as e:
#         print('Issue with image {}'.format(image_path))

#building data pipeline batch size initialized to 36 and initialized to 256x256
data = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size = 16, image_size = (100, 500)) #here u can change the size of it
data_iterator = data.as_numpy_iterator()
print(len(data))

#BATCH EXAMPLE
# class 0 = fraud, class 1 = non fraud
batch = data_iterator.next()

# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])

scaled = batch[0]/255
print(scaled.max())

#PREPROCESSING
# x represent images and y represent lables, loading batches and scaling down
data = data.map(lambda x,y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
# # example
batch = scaled_iterator.next()
print(batch[0].max())
# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img)
#     ax[idx].title.set_text(batch[1][idx])

#should equal data size
train_size = int(len(data)*.7) #maybe change to 0.8
val_size = int(len(data)*.2)
test_size = int(len(data)*.1 +1)
print("data size: ", len(data))
print("split size: ", test_size+train_size+val_size)

print(train_size)
print(val_size)
print(test_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#BUILDING MODEL
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(8, (3,3), 1, activation='relu', input_shape=(100,500,3)))#16
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))#32
model.add(MaxPooling2D())
model.add(Conv2D(8, (3,3), 1, activation='relu'))#16
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(600, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()

# #TRAINING
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


#PLOTTING LOSS
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
# plt.show()

#PLOTTING ACCURACY
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
# plt.show()


# ############# EVALUATION #############
from keras.metrics import Precision, Recall, BinaryAccuracy
# from tensorflow.keras.metrics import Prexcision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

# print(pre.result(), re.result(), acc.result())

path = "dataset/ML/unscanned uncropped checks" ##### RELATIVE PATH
unscannedUncroppedList = os.listdir(path)
unscannedUncroppedList = sorted(unscannedUncroppedList)
print("Number of unscannedUncroppedList: ", len(unscannedUncroppedList))
# print(unscannedUncroppedList)

count = 0

# unscanned checks
for originalImg in unscannedUncroppedList:

  # making sure we're checking the right files
  if originalImg.__contains__(".jpg"):

    # image path
    checkImgPath = "dataset/ML/unscanned uncropped checks/" + originalImg ##### RELATIVE PATH
    img = cv2.imread(checkImgPath)

    # resizing image
    img = cv2.resize(img, (900, 450))

    # cropping and displaying image
    if int(originalImg[6]) == 1:
      cropped_image = img[110:200, 110:660] # for type 1

    elif int(originalImg[6]) == 2:
      cropped_image = img[80:200, 130:545] # for type 2

    elif int(originalImg[6]) == 3:
      cropped_image = img[115:220, 135:630] # for type 3

    elif int(originalImg[6]) == 4:
      cropped_image = img[100:210, 100:680] # for type 4

    else:
      count +=1
      continue

    cropped_image = cv2.resize(cropped_image, (500, 100))

    # display cropped image
    # cv2_imshow(cropped_image)

    #saving image
    croppedFilePath = "dataset/ML/unscanned checks/" + originalImg ##### RELATIVE PATH
    cv2.imwrite(croppedFilePath, cropped_image)

# Base directory for outputs
outputFolder = 'dataset/ML/model output' ##### RELATIVE PATH

# Sub-directory paths within 'model_output'  
NonFraudFolder = outputFolder + "/non fraud" ##### RELATIVE PATH
FraudFolder = outputFolder + "/fraud" ##### RELATIVE PATH
# manualCheckingFolder = os.path.join(outputFolder, 'manual checking')

# Ensure the base output directory exists
os.makedirs(outputFolder, exist_ok=True)

# Ensure the subdirectories within 'model_output' exist
os.makedirs(NonFraudFolder, exist_ok=True)
os.makedirs(FraudFolder, exist_ok=True)
# os.makedirs(manualCheckingFolder, exist_ok=True)

inputFolder = 'dataset/ML/unscanned checks' ##### RELATIVE PATH
image_exts = ['jpeg','jpg', 'bmp', 'png']
check_images = [img for img in os.listdir(inputFolder) if img.lower().endswith(tuple(image_exts))]


#PREDICTING 
thresholdForManualChecking = 0.1  # adjust as needed

for check in check_images:
    # Read and preprocess
    img_path = os.path.join(inputFolder, check)
    img = cv2.imread(img_path)
    resized_img = tf.image.resize(img, (100,500))
    scaled_img = resized_img / 255.0

    # Predict
    prediction = model.predict(np.expand_dims(scaled_img, 0))

    # Determine sub-folder based on prediction
    if prediction > 0.999:
        destination_folder = NonFraudFolder
    else:
        destination_folder = FraudFolder
    # else:
    #     destination_folder = manualCheckingFolder

    # Move image to the corresponding sub-folder
    destination_path = os.path.join(destination_folder, check)
    shutil.copyfile(img_path, destination_path)


#CONFUSION MATRIX
counter = 0
TP, FP, TN, FN = 0, 0, 0, 0
path = "dataset/ML/model output/fraud" ##### RELATIVE PATH
fraudList = os.listdir(path)
fraudList = sorted(fraudList)
# print("fraudList: ", len(fraudList))
for check in fraudList:
  counter += 1
  if check.__contains__("F"):
    TP +=1
  elif check.__contains__("N"):
    FP +=1

path = "dataset/ML/model output/non fraud" ##### RELATIVE PATH
nonFraudList = os.listdir(path)
nonFraudList = sorted(nonFraudList)
# print("nonFraudList: ", len(nonFraudList))
for check in nonFraudList:
  counter += 1
  if check.__contains__("N"):
    TN +=1
  elif check.__contains__("F"):
    FN +=1

print("count", counter, "| TP", TP, "| TN", TN, "| FP", FP, "| FN", FN)
print("Percent TP: %", (100*(TP/counter)))
print("Percent TN: %", (100*(TN/counter)))
print("Percent FP: %", (100*(FP/counter)))
print("Percent FN: %", (100*(FN/counter)))