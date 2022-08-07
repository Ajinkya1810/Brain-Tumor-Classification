# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:35:37 2022

@author: Dell
"""
# Python3 code to remove whitespace
def remove(string):
	return "".join(string.split())
	
# Driver Program
string = 'gg (1)_PPNZ00P.jpg'
print(remove(string))

