import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data_list =[]
for i in range(0,8):
    data_list.append(np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32))

#data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

images_list = []
# Replace this with the path to your image
images_list.append(Image.open('E:/Work/ITI/AI/fruits-360/Test/banana.jpg'))
images_list.append(Image.open('E:/Work/ITI/AI/fruits-360/Test/banana2.jpg'))
images_list.append(Image.open('E:/Work/ITI/AI/fruits-360/Test/Bananas.jpg'))
images_list.append(Image.open('E:/Work/ITI/AI/fruits-360/Test/cherry.jpg'))
images_list.append(Image.open('E:/Work/ITI/AI/fruits-360/Test/strawberry.jpg'))
images_list.append(Image.open('E:/Work/ITI/AI/fruits-360/Test/cherries.jpg'))
images_list.append(Image.open('E:/Work/ITI/AI/fruits-360/Test/cherries_strawberries.jpg'))
images_list.append(Image.open('E:/Work/ITI/AI/fruits-360/Test/watermelon.jpg'))

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
idx=0
for img in images_list:
    img = ImageOps.fit(img, size, Image.ANTIALIAS)
    images_list[idx] = img
    idx +=1

images_array_list=[]
#turn the image into a numpy array
for img in images_list:
    images_array_list.append( np.asarray(img) )

# display the resized image
#image.show()
normalized_image_array=[]
for image_arr in images_array_list:
    # Normalize the image
    normalized_image_array.append((image_arr.astype(np.float32) / 127.0) - 1)

# Load the image into the array
data_idx =0
for norm_image_arr in normalized_image_array:
    data_list[data_idx][0] = norm_image_arr 
    data_idx +=1
#data[0] = norm_image_arr

# run the inference
for data in data_list:
    prediction = model.predict(data)
    #print(prediction)
    prediction_list = list(prediction[0])

    classes = ['Mango','Banana','Strawberry','Watermelon','cherry']
    class_index = prediction_list.index(max(prediction_list))
    print(classes[class_index])
