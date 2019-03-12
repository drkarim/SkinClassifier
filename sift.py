import cv2
import numpy as np
import glob
import os



for image in glob.glob('D:/University/Y3/Dissertation/SIFT/skin-cancer-mnist-ham10000/HAM10000_images_part_1/*'):
    print(image)
    
    
    img = cv2.imread(image)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    
    #img=cv2.drawKeypoints(gray,kp)
    img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imwrite(("D:/University/Y3/Dissertation/SIFT/sift_data/Data_part1/{}").format(os.path.basename(image)),img)
    
    
for image in glob.glob('D:/University/Y3/Dissertation/SIFT/skin-cancer-mnist-ham10000/HAM10000_images_part_2/*'):
    print(image)
    
    
    img = cv2.imread(image)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    
    #img=cv2.drawKeypoints(gray,kp)
    img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imwrite(("D:/University/Y3/Dissertation/SIFT/sift_data/Data_part2/{}").format(os.path.basename(image)),img)    