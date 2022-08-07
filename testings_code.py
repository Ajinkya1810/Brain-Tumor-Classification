# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:58:33 2022

@author: Dell
"""
#######  Test New Image
from keras.models import load_model
import numpy as np
from keras.preprocessing import image


# load the model we saved
model = load_model('trained_model.h5')

# predicting New images
test_image = image.load_img('gg (1).jpg', target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

#train_set.class_indices
if result[0][0] >0.90: 
    print('Glioma Image')
elif result[0][1]  >0.90: 
    print('Meningioma Image')
elif result[0][2]  >0.90: 
    print('Normal Image')
else:
    print("pituatary image")
