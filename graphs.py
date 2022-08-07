# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 19:10:28 2022

@author: Dell
"""
import matplotlib.pyplot as plt ### visualisation
plt.plot(history_model.history['accuracy'])
plt.plot(history_model.history['val_accuracy'])
plt.legend(['accuracy','val_accuracy'])
plt.show()

# Loss Comparison
plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show()