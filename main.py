import numpy as np
from fastapi import  FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Model = tf.keras.models.load_model("./Model/1")
Class_name = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello"


def read_file_as_image(data)->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)

):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predications = Model.predict(img_batch)
    Output = Class_name[np.argmax(predications[0])]
    Confidence = np.max(predications[0])
    return {
        'class': Output,
        'Confidence': float(Confidence)*100
    }

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)