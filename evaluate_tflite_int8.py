# Install tflite_runtime package to evaluate the model.
#!pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl

# Now we do evaluation on the tflite model.
import os
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from PIL import Image
from PIL import ImageDraw
#%matplotlib inline

detect_path='./test_img'
save_path='./saved_detect'
# Creates tflite interpreter
#interpreter = Interpreter('./model_float32.tflite')
# This exact code can be used to run inference on the edgetpu by simply creating 
# the instantialize the interpreter with libedgetpu delegates:
interpreter = Interpreter('./model_edgetpu_int8.tflite', experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()
interpreter.invoke() # warmup
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
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

#test_image_paths = [os.path.join('./test_img', 'image{}.jpg'.format(i)) for i in range(1, 6)]
img_dir = os.listdir(detect_path)
for image_name in img_dir:
  image_path=os.path.join(detect_path, image_name)
  image = Image.open(image_path)
  image_width, image_height = image.size
  draw = ImageDraw.Draw(image)
  resized_image = image.resize((width, height))
#  np_image = np.asarray(resized_image)
  np_image = np.asarray(resized_image,dtype=np.float32)
  
  np_image=(2.0 / 255.0) * np_image - 1.0
#  np_image = np.array(image).reshape(image_height, image_width, 3).astype(np.uint8)
  input_tensor = np.expand_dims(np_image, axis=0)
  # Run inference
  boxes, classes, scores = run_inference(interpreter, input_tensor)
  # Draw results on image
  colors = {0:(128, 255, 102), 1:(102, 255, 255)}
  labels = {0:'bee', 1:'bee'}
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

#  display(image)
  print('Evaluating: %s , %d object detected.' % (image_path, detect_number) )
  image.save(save_path + '/' + image_name)
#  image.show()
