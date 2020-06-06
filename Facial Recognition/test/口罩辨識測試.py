# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:17:02 2020

@author: allen
"""
import cv2

def main():
    
    image = cv2.imread('test.jpg')
    image2 = cv2.imread('test2.jpg')
    size = image.shape

    #圖片切半
    image = image[int(size[1]/2):size[1],0:size[0]]
    image2 = image2[int(size[1]/2):size[1],0:size[0]]
    #HSV = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("image",image)
    cv2.imshow("image2",image2)
    #cv2.imshow("HSV",HSV)
    #cv2.imshow("gray",gray)
    
    size = image.shape
    #print(size[0],size[1])
    
    #print("image[42][91][0] = ",image[42][91][0])
    #print("image2[42][91][0] = ",image2[42][91][0])
    
    #image[y][x][BGR]
    # 注意 順序是BGR
    pixel=0
    
    total = 0    
    # y = 0~114
    for i in range(0,size[0]):
        # x = 0~228
        for j in range(0,size[1]):
            
            #偵測藍色範圍
            if image[i][j][0] >= 160 and image[i][j][0] <= 230:
                pixel += 1

            total += 1
                
    pixel2 = 0
    total = 0    
    # y = 0~114
    for i in range(0,size[0]):
        # x = 0~228
        for j in range(0,size[1]):
            #順序是BGR!!
            #偵測藍色範圍
            if image2[i][j][0] >= 160 and image2[i][j][0] <= 230:
                pixel2 += 1

            total += 1

    print("pixel = ",pixel)
    print("pixel2 = ",pixel2)
    print("total = ",total)
    
    if pixel > total * 0.65:
        print("有戴口罩")
    
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()