from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import re
import math


# app = FastAPI(docs_url="/docs", root_path="/ear_keypoints")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_trained_model():
    model_2ear = load_model("./weights/bestmodel_2ears.h5", compile=False)
    model_1ear = load_model("./weights/bestmodel_1ear.h5", compile=False)
    classmodel = load_model(
        "./weights/bestmodel_classification.h5", compile=False)
    return model_2ear, model_1ear, classmodel


model1, model2, classmodel = load_trained_model()


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


def preprocessing(im):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(im)
        if not results.multi_face_landmarks:
            # couldn't find whole face in the image
            params = [0, im.shape[1], 0, im.shape[0]]
            im = cv2.resize(im, (256, 256))

        else:
            offset = [12, 8, 12, 12]  # top bottom left right
            points = [10, 234, 454, 152]

            top_offset = offset[0]
            bot_offset = offset[1]
            left_offset = offset[2]
            right_offset = offset[3]
            x_points = []
            y_points = []
            for r in points:
                x = results.multi_face_landmarks[0].landmark[r].x
                y = results.multi_face_landmarks[0].landmark[r].y
                shape = im.shape
                relative_x = int(x * shape[1])
                x_points.append(relative_x)
                relative_y = int(y * shape[0])
                y_points.append(relative_y)

            ytop = min(y_points)
            ybot = max(y_points)
            xtop = min(x_points)
            xbot = max(x_points)

            ytop = max(0, int(ytop - abs(ytop - ybot) * top_offset / 100))
            ybot = min(im.shape[0], int(
                ybot + abs(ytop - ybot) * bot_offset / 100))
            xtop = max(0, int(xtop - abs(xtop - xbot) * left_offset / 100))
            xbot = min(im.shape[1], int(
                xbot + abs(xtop - xbot) * right_offset / 100))

            im = im[ytop:ybot, xtop:xbot]
            params = [xtop, xbot, ytop, ybot]
            return cv2.resize(im, (256, 256))


def grayscale(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def getyaw(im):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(im)
        yaw1 = math.atan((results.multi_face_landmarks[0].landmark[234].z - results.multi_face_landmarks[0].landmark[454].z) / (
            results.multi_face_landmarks[0].landmark[234].x - results.multi_face_landmarks[0].landmark[454].x))
        yaw2 = math.atan((results.multi_face_landmarks[0].landmark[132].z - results.multi_face_landmarks[0].landmark[361].z) / (
            results.multi_face_landmarks[0].landmark[132].x - results.multi_face_landmarks[0].landmark[361].x))
        yaw3 = math.atan((results.multi_face_landmarks[0].landmark[162].z - results.multi_face_landmarks[0].landmark[389].z) / (
            results.multi_face_landmarks[0].landmark[162].x - results.multi_face_landmarks[0].landmark[389].x))
    return (abs(yaw1+yaw2+yaw3)/3)


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def rotateandcrop(img):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(img)
        if results.multi_face_landmarks:
            annotated_image = img.copy()
            im_shape = annotated_image.shape
            x_diff = abs(results.multi_face_landmarks[0].landmark[234].x -
                         results.multi_face_landmarks[0].landmark[454].x)  # Roll
            y_diff = (results.multi_face_landmarks[0].landmark[234].y -
                      results.multi_face_landmarks[0].landmark[454].y)
            roll = (y_diff/x_diff)
            theta = math.degrees(math.atan(roll))
            c_x = (
                results.multi_face_landmarks[0].landmark[234].x) * im_shape[1]
            c_y = (
                results.multi_face_landmarks[0].landmark[234].y) * im_shape[0]

            rotated = rotate(img, -theta, (c_x, c_y))

            results = face_mesh.process(rotated)

            # Print and draw face mesh landmarks on the image.
            if results.multi_face_landmarks:
                annotated_image = rotated.copy()
                im_shape = rotated.shape

                top_left_x = int(
                    (results.multi_face_landmarks[0].landmark[234].x)*im_shape[1] - 0.2*im_shape[1])
                top_left_y = int(
                    (results.multi_face_landmarks[0].landmark[10].y)*im_shape[0] - 0.1*im_shape[0])

                bottom_right_x = int(
                    (results.multi_face_landmarks[0].landmark[454].x) * im_shape[1] + 0.2*im_shape[1])
                bottom_right_y = int(
                    (results.multi_face_landmarks[0].landmark[152].y) * im_shape[0] + 0.05*im_shape[0])

                if top_left_x < 0:
                    top_left_x = 0
                if top_left_y < 0:
                    top_left_y = 0
                if bottom_right_x > im_shape[1]:
                    bottom_right_x = im_shape[1]
                if bottom_right_y > im_shape[0]:
                    bottom_right_y = im_shape[0]

                crop = annotated_image[top_left_y:bottom_right_y,
                                       top_left_x:bottom_right_x]
                return crop


def base64_to_pil(img_base64):
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def image_to_base64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    return str(jpg_as_text)[2:-1]


@app.get("/")
async def ping():
    return "Hello, I am alive"


@app.post("/showcroprotate")
async def show_image(file: UploadFile = File(...)):
    image_orig = read_file_as_image(await file.read())
    image = rotateandcrop(image_orig)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("out.png", image)
    return FileResponse("out.png")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_orig = read_file_as_image(await file.read())
    image_rc = rotateandcrop(image_orig)
    p = getyaw(image_rc)
    if getyaw(image_rc) < 0.29:
        image_color = preprocessing(image_orig)
        image = grayscale(image_color)
        image_batch = np.expand_dims(image, 0)
        image_batch = np.expand_dims(image_batch, -1)
        classification = classmodel.predict(np.expand_dims(image_color, 0))
        predictions = model1.predict(image_batch)
        prob_le = classification[0][0]
        prob_re = classification[0][1]
        lex = predictions[0][2]
        ley = predictions[0][3]
        rex = predictions[0][4]
        rey = predictions[0][5]
    else:
        image = cv2.resize(image_rc, (512, 512))
        image_batch = np.expand_dims(image, 0)
        preds = model2.predict(image_batch)
        prob_le = "NA"
        prob_re = "NA"
        lex = preds[0][0]
        ley = preds[0][1]
        rex = preds[0][0]
        rey = preds[0][1]

    return {
        "left_ear_prob": str(prob_le),
        "right_ear_prob": str(prob_re),
        "left_ear_x": str(lex),
        "left_ear_y": str(ley),
        "right_ear_x": str(rex),
        "right_ear_y": str(rey),
        "p": str(p)
    }


@app.post("/predictandshow")
async def show_image(file: UploadFile = File(...)):
    image_orig = read_file_as_image(await file.read())
    image_orig_w, image_orig_h = image_orig.shape[1], image_orig.shape[0]
    radius = 3  # int(max(image_orig_w, image_orig_h)/200)
    image_rc = rotateandcrop(image_orig)
    if getyaw(image_rc) < 0.29:
        image_color = preprocessing(image_orig)
        image = grayscale(image_color)
        image_batch = np.expand_dims(image, 0)
        image_batch = np.expand_dims(image_batch, -1)
        classification = classmodel.predict(np.expand_dims(image_color, 0))
        predictions = model1.predict(image_batch)
        prob_le = classification[0][0]
        prob_re = classification[0][1]
        lex = predictions[0][2]
        ley = predictions[0][3]
        rex = predictions[0][4]
        rey = predictions[0][5]
        image = image_color
        if prob_le > 0.9:
            image = cv2.circle(image, (int(lex), int(ley)),
                               radius, (255, 0, 0), -1)
        else:
            image = cv2.circle(image, (int(lex), int(ley)),
                               radius, (0, 255, 0), -1)
        if prob_re > 0.9:
            image = cv2.circle(image, (int(rex), int(rey)),
                               radius, (255, 0, 0), -1)
        else:
            image = cv2.circle(image, (int(rex), int(rey)),
                               radius, (0, 255, 0), -1)

    else:
        image = cv2.resize(image_rc, (512, 512))
        image_batch = np.expand_dims(image, 0)
        preds = model2.predict(image_batch)
        prob_le = "NA"
        prob_re = "NA"
        lex = preds[0][0]
        ley = preds[0][1]
        rex = preds[0][0]
        rey = preds[0][1]
        image = cv2.circle(
            image, (int(preds[0][0]), int(preds[0][1])), 5, (255, 0, 0), -1)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("out.png", image)
    return FileResponse("out.png")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3000)
