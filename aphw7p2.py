import matplotlib.pyplot as plt
import zipfile
import glob
import os
import numpy as np

import random
import cv2
from numpy import linalg as LA
#from PIL import Image, ImageDraw
#from PIL import ImageFont
"""
def play(silent=False):
    M=generateArray()
    #making an RGB image
    img = Image.new('RGB', (1000, 1008), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    #specifying the font size
    font = ImageFont.truetype(r"arial.ttf", 40)
    #drawing a rectangle for the table
    d.rectangle((0, 91, 1000, 1000), fill=(255, 255, 204), outline=(255, 255, 204))
    #writing my name
    d.text((200, 20), "Mohammad Javad Rajbar 9523048 ", font=font, fill=(0, 0, 0))
    temp = 0
    # to make those dashed line we use these fors
    for i in range(80):
        for j in range(80):
            d.line((temp, 91 * (i + 1), 9 + temp, 91 * (i + 1)), fill=(255, 0, 0), width=3)
            temp = temp + 20
        temp = 0
    temp = 90
    for i in range(50):
        for j in range(50):
            temp = temp + 20
            d.line((i * 100, temp, i * 100, temp + 9), fill=(255, 0, 0), width=3)
        temp = 90
    font = ImageFont.truetype(r"arial.ttf", 15)
    #and we write data in the table
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            d.text((100 * i + 20, 90 * j + 135), " " + str((10*j + i +1)) + " : " + str(M[i, j]) + " ", font=font,
                   fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    #showing and saving
    if (silent == False):
        img.save('Table.png')
        img=cv2.imread('Table.png')
        cv2.imshow('Table', img)
        cv2.waitKey()
    else:
        img.save('Table.png')
    return M,M[0][8]
    """
def generateArray():
    #Making an 10x10 array with random number between 100 and 999
    M = np.random.randint(100, 999, (10, 10))
    for i in range(M.shape[1]):
        for j in range(M.shape[0]):
            if (i + j == 8):
                #specifying the wanted ones
                M[i][j] = 632
    return M
def play (silent=False):

    M = generateArray()
    #making a table 
    plt.title("MohammadJavadRanjbar 9523048", fontsize=15, loc='center')
    M=np.array(M)
    data= np.array2string(M)
    temp = 1
    data = []
    for i in range(M.shape[0]):
        T_vec = []
        for j in range(M.shape[1]):
            T_vec.append(str(temp)+" : " +str( M[i][j]))
            temp += 1
        data.append(T_vec)
    #set colors name 
    rowColours = ['cyan', 'red', 'darkblue', 'darkred', 'pink', 'orange', 'purple', 'blue','black','yellow','white']
    The_Table = plt.table(cellText=data, loc='lower center',cellLoc='center')
    The_Table.set_fontsize(12)
    The_Table.scale(1.2, 2)
    plt.axis('off')
    #seting every cell settings
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            The_Table[i, j].set_linestyle('dashed')
            The_Table[(i, j)].set_facecolor('#EEE8AA')
            The_Table[(i, j)].set_linestyle('dashed')
            The_Table[(i, j)].get_text().set_color(rowColours[random.randint(0, 9)])
            The_Table[(i, j)].set_edgecolor('red')
    if (silent == False):
        plt.savefig("table.png")
        img = cv2.imread('table.png')
        cv2.imshow('img', img)
        cv2.waitKey()

    else:
        img = cv2.imread('table.png')

    return M,M[0][8]