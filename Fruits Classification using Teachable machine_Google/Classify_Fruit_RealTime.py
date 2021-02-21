import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
  
# in readme file 1- introduction what you are doing ?? 
#                2- installation --> packages required ?? and so on .. 
#                3- How to run ?? 
#
# if you want to talk to the bash of the google colab use ! or % then your command 
# 

# define a video capture object 
vid = cv2.VideoCapture(0) 

# Load the model
model = tensorflow.keras.models.load_model(r'C:\Users\abdoo\AppData\Local\Programs\Python\Python38\Lib\site-packages\tensorflow\python\saved_model\keras_model.h5')

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)


    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)
    #image = ImageOps.fit(frame, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
   # image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    #print(normalized_image_array)

    # Load the image into the array
    data[0] = normalized_image_array

    #print(data)
    # run the inference
    prediction = model.predict(data)
   # print(prediction)

    prediction_list = list(prediction[0])

    classes = ['Mango','Banana','Strawberry','Watermelon','cherry']
    class_index = prediction_list.index(max(prediction_list))
    print(classes[class_index])


# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() # import the opencv library 
