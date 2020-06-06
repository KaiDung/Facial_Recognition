import cv2

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

x = 200
y = 160
length = 228

def main():
    cap = cv2.VideoCapture(0)
    
    while(1):
        
        ret,frame = cap.read()
        
        #戴著口罩haar會抓不到人臉
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            test = frame[y:y+h,x:x+w]
        '''
        
        cv2.rectangle(frame,(x,y),(x+length,y+length),(0,255,0),2)
        test = frame[y:y+length,x:x+length]
        frame = cv2.flip(frame,2)
        cv2.imshow("frame",frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("test2.jpg",test)
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()