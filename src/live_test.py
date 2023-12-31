import cv2
import torch
import torchvision.transforms as transforms
from model import *
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN128()
model.to(DEVICE)
model.load_state_dict(torch.load("model/model_1"))
model.eval()

SHAPE = (128,128)
emotion_map = {0.:"angry", 1.:"happy", 2.:"sad", 3.:"shocked"}

def preprocess(frame, shape, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, shape)

    frame = torch.tensor(frame).to(device).view(1,1,*shape).float() / 255

    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        frame_shape = frame.shape[:-1]


        width = int(min(frame_shape) / 2)

        x = int(frame_shape[0]/2 - width/2)
        y = int(frame_shape[1]/2 - width/2)
        croped = frame[x:x+width, y:y+width]

        cv2.line(frame, (0, x), (frame_shape[1], x), (255, 0, 0), 1)
        cv2.line(frame, (0, x+width), (frame_shape[1], x+width), (255, 0, 0), 1)

        cv2.line(frame, (y, 0), (y, frame_shape[0]), (255, 0, 0), 1)
        cv2.line(frame, (y+width, 0), (y+width, frame_shape[0]), (255, 0, 0), 1)


        data = preprocess(croped, SHAPE, DEVICE)

        logits = model(data)
        prediction = torch.argmax(logits, axis = 1).item()

        cv2.putText(frame, f"Emotion detected: {emotion_map[prediction]}", (y+5, x-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)



        cv2.imshow("Camera Feed", frame)
        cv2.imshow("data", data.cpu().numpy().reshape(128,128))
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    cap.release()
    cv2.destroyAllWindows()
