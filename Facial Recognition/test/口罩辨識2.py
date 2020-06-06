# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 19:27:57 2020

@author: allen
"""
import cv2

x = 200
y = 160
length = 228

global OK
OK = 0
def main():
    cap = cv2.VideoCapture(0)
    
    while(1):
        global OK
        ret,frame = cap.read()
        frame = cv2.flip(frame,2)
        face = frame[y:y+length,x:x+length]
        
        #臉切半
        size = face.shape
        half = face[int(size[1]/2):size[1],0:size[0]]
        size = half.shape
        
        #轉灰階
        gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
        #二值化
        ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        
        cv2.imshow("Binary",thresh)
        
        pixel = 0
        #計算白色面積
        # x = 0~228
        for i in range(0,thresh.shape[0]):
            # y = 0~114
            for j in range(0,thresh.shape[1]):
                if thresh[i][j] == 255:
                    pixel += 1
        
        if pixel >= size[0]*size[1]*0.7:
            cv2.rectangle(frame,(x,y),(x+length,y+length),(0,255,0),2)
            cv2.imshow("frame",frame)
            print("有戴口罩")
        else:
            cv2.rectangle(frame,(x,y),(x+length,y+length),(0,0,255),2)
            cv2.imshow("frame",frame)
            print("沒戴口罩")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()