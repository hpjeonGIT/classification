import sys
import argparse
import numpy as np
from PIL import Image
import json
from keras.preprocessing import image
from keras.models import model_from_json
# Ref: https://github.com/DeepLearningSandbox/DeepLearningSandbox/tree/master/image_recognition
target_size = (224,224)
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model_second.h5')

def predict(model, img, target_size):
  if img.size != target_size:
    img = img.resize(target_size)
  x = image.img_to_array(img)
  x = np.expand_dims(x,axis=0)
  y_prob = model.predict(x)
  for i in range(len(y_prob[0])):
    print('Label = ', rev[i], ':', '%5.3f'%(100*y_prob[0][i]))
    
if __name__ == '__main__':
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  args = a.parse_args()
  global rev
  try:
    labels = json.load(open('labels.txt'))
  except IOError:
    print("labels.txt NOT found")
    sys.exit()
    
  rev = dict((vk,k) for k , v in zip(labels.keys(), labels.values()))
  
  if args.image is None:
    a.print_help()
    sys.exit(1)
    
  if args.image is not None:
    img = Image.open(args.image)
    predict(model, img, target_size)
