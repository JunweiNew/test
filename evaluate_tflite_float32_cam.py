#!pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl

import os
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from PIL import Image
from PIL import ImageDraw
import cv2


interpreter = Interpreter('./model_float32.tflite')
#interpreter = Interpreter('./model_float32.tflite', experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()
interpreter.invoke() # warmup
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))


image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


def run_inference(interpreter, image):
  interpreter.set_tensor(input_details[0]['index'], image)
  interpreter.invoke()
  boxes = interpreter.get_tensor(output_details[0]['index'])[0]
  classes = interpreter.get_tensor(output_details[1]['index'])[0]
  scores = interpreter.get_tensor(output_details[2]['index'])[0]
  # num_detections = interpreter.get_tensor(output_details[3]['index'])[0]
  return boxes, classes, scores


while(True):
  ret, frame = cap.read()
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame_resized = cv2.resize(frame_rgb, (width, height)).astype(np.float32)
  frame_resized = (2.0 / 255.0) * frame_resized - 1.0  #quantization -1 to 1

  input_tensor = np.expand_dims(frame_resized, axis=0)
  boxes, classes, scores = run_inference(interpreter, input_tensor)
  colors = {0:(128, 255, 102), 1:(102, 255, 255)}
  labels = {0:'bee'}

  for i in range(len(boxes)):
    if scores[i] > .6:
      ymin = int(max(1, (boxes[i][0] * image_height)))
      xmin = int(max(1, (boxes[i][1] * image_width)))
      ymax = int(min(image_height, (boxes[i][2] * image_height)))
      xmax = int(min(image_width, (boxes[i][3] * image_width)))
     # cv2..rectangle((xmin, ymin, xmax, ymax), width=7, outline=colors[int(classes[i])])
      cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)
     # cv2.rectangle((xmin, ymin, xmax, ymin-10), fill=colors[int(classes[i])])
      text = labels[int(classes[i])] + ' ' + str(round(scores[i]*100,2)) + '%'
     # cv2.putText((xmin+2, ymin-10), text, fill=(0,0,0), width=2)

      cv2.putText(frame, text, (xmin, ymin-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


 # print('Evaluating: %s , %d object detected.' % (image_path, detect_number) )
 # image.save(save_path + '/' + image_name)
  cv2.imshow('Object detector', frame)

  #if cv2.waitKey(1) == ord('q'):
      #break
  if cv2.waitKey(5) & 0xFF == 27:
    break

cap.release()
cv2.destroyAllWindows()

