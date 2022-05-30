# from matplotlib import image
import streamlit as st
from PIL import Image
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np





# Load Images
@st.cache
def load_image(img_file):
    img = Image.open(img_file)
    return img


st.title('Pothole Detection')
st.subheader('By Arjun M S')

img = st.file_uploader('Upload an Image file',type = ['png','jpeg','jpg'])
if img is not None :
    # see details
    # st.write({"Filename":img.name})
    st.image(load_image(img))
    if st.button("Predict"):
        model = load_model('keras_model.h5')
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Replace this with the path to your image
        image = Image.open(img)
        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        #turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        # run the inference
        result = model.predict(data)
        print(result)
        if result[0][0] == 1:
            prediction = 'pothole'
        else:
            prediction = 'normal'
        st.write(f"Prediction: {prediction}")
