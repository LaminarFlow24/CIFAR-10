import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

st.title("Object recognizer using CIFAR-10 Dataset")
st.write("Made by Yashas Jain")
liss = ['Plane','Bird','Car','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to 32x32x3 array
    resized_image = image.resize((32, 32))
    image_array = np.array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    if image_array.shape == (1, 32, 32, 3):  # Check the shape with batch dimension
        st.write("The image has been successfully converted to a 32x32x3 array.")
        
        # Load the deep learning model
        loaded_model = load_model('model_aug.h5')

        # Normalize pixel values (assuming your model expects normalized input)
        image_array = image_array / 255.0

        # Predict with the loaded model
        result = loaded_model.predict(image_array)[0]
        maxx  = 0
        indexx = 0
        
        for i in range(len(result)):
            if maxx < result[i]:
                maxx = result[i]
                indexx = i

        st.write(liss[indexx])
    else:
        st.write(f"The image array shape is: {image_array.shape}. It may not be a color image.")

