## Imports ##
from fastapi import FastAPI, File, UploadFile

import numpy as np
from io import BytesIO
from PIL import Image

from skimage.color import rgb2gray
from skimage import transform

from tensorflow.keras.models import load_model

## Setup ##
app = FastAPI()

## Main ##
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/handcoded/mnist-cnn", responses={200: {"description": "Returns a json with a prediction of the class"}})
async def handcoded_mnist_cnn(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.array(Image.open(BytesIO(contents)))

    model = load_model('models/mnist-cnn')

    image = np.array(image)
    image = rgb2gray(image)

    image = transform.resize(image, (28,28), mode='constant', anti_aliasing=True)

    image = image.reshape((1, 28, 28, 1))

    print(image.shape)

    predict = model.predict_classes(image)

    return {"Class": str(predict[0])}

@app.post("/autokeras/mnist-cnn", responses={200: {"description": "Returns a json with a prediction of the class"}})
async def handcoded_mnist_cnn(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.array(Image.open(BytesIO(contents)))

    model = load_model('models/mnist-cnn')

    image = np.array(image)
    image = rgb2gray(image)

    image = transform.resize(image, (28,28), mode='constant', anti_aliasing=True)

    image = image.reshape((1, 28, 28, 1))

    print(image.shape)

    predict = model.predict_classes(image)

    return {"Class": str(predict[0])}