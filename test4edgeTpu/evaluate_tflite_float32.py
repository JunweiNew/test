#!pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl

import os
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from PIL import Image
from PIL import ImageDraw

detect_path='../test_img'
save_path='./saved_detect'
interpreter = Interpreter('./model_float32.tflite')
#interpreter = Interpreter('./model_float32.tflite', experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()
interpreter.invoke() # warmup
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("input_details:", str(input_details))
print("output_details:", str(output_details))
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]


def run_inference(interpreter, image):
  interpreter.set_tensor(input_details[0]['index'], image)
  interpreter.invoke()
  boxes = interpreter.get_tensor(output_details[0]['index'])[0]
  classes = interpreter.get_tensor(output_details[1]['index'])[0]
  scores = interpreter.get_tensor(output_details[2]['index'])[0]
  # num_detections = interpreter.get_tensor(output_details[3]['index'])[0]
  return boxes, classes, scores


img_dir = os.listdir(detect_path)
for image_name in img_dir:
  image_path=os.path.join(detect_path, image_name)
  image = Image.open(image_path)
  image_width, image_height = image.size
  draw = ImageDraw.Draw(image)
  resized_image = image.resize((width, height))
  np_image = np.asarray(resized_image,dtype=np.float32)

  #quantization -1 to 1
  np_image=(2.0 / 255.0) * np_image - 1.0

  input_tensor = np.expand_dims(np_image, axis=0)
  boxes, classes, scores = run_inference(interpreter, input_tensor)
  colors = {0:(128, 255, 102), 1:(102, 255, 255)}
  labels = {0:'bee'}
  detect_number = 0
  for i in range(len(boxes)):
    if scores[i] > .6:
      ymin = int(max(1, (boxes[i][0] * image_height)))
      xmin = int(max(1, (boxes[i][1] * image_width)))
      ymax = int(min(image_height, (boxes[i][2] * image_height)))
      xmax = int(min(image_width, (boxes[i][3] * image_width)))
      draw.rectangle((xmin, ymin, xmax, ymax), width=7, outline=colors[int(classes[i])])
      draw.rectangle((xmin, ymin, xmax, ymin-10), fill=colors[int(classes[i])])
      text = labels[int(classes[i])] + ' ' + str(round(scores[i]*100,2)) + '%'
      draw.text((xmin+2, ymin-10), text, fill=(0,0,0), width=2)
      detect_number += 1


  print('Evaluating: %s , %d object detected.' % (image_path, detect_number) )
  image.save(save_path + '/' + image_name)





