# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:30:41 2020

@author: allen
"""
import cv2

x = 200
y = 160
length = 228

def main():
    cap = cv2.VideoCapture(0)
    
    while(1):
        
        ret,frame = cap.read()
        
        face = frame[y:y+length,x:x+length]
        
        #臉切半
        size = face.shape
        half = face[int(size[1]/2):size[1],0:size[0]]
        size = half.shape
        
        #image[y][x][BGR]
        # 注意 順序是BGR
        pixel=0
        
        total = 0    
        # y = 0~114
        for i in range(0,size[0]):
            # x = 0~228
            for j in range(0,size[1]):
                #偵測藍色範圍
                if half[i][j][0] >= 160 and half[i][j][0] <= 230:
                    pixel += 1
                total += 1
                
        if pixel > total * 0.65:
            cv2.rectangle(frame,(x,y),(x+length,y+length),(0,255,0),2)
            frame = cv2.flip(frame,2)
            cv2.imshow("frame",frame)
        else:
            cv2.rectangle(frame,(x,y),(x+length,y+length),(0,0,255),2)
            frame = cv2.flip(frame,2)
            cv2.imshow("frame",frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()