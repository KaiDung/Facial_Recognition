import smtplib
from email.mime.multipart import MIMEMultipart #email內容載體
from email.mime.text import MIMEText #用於製作文字內文
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase #用於承載附檔
from email import encoders #用於附檔編碼
import datetime
import ssl
from pathlib import Path

def SEND(to_address,image):
    
    #寄件者使用的Gmail帳戶資訊
    gmail_user = '406262175@gapp.fju.edu.tw'
    gmail_password = 'rifdqhgxqtrjkudo'
    from_address = gmail_user
      
    #設定信件內容與收件人資訊  
    Subject = "Subject"
    contents = "this is content"
    
    #開始組合信件內容
    mail = MIMEMultipart()
    mail['From'] = from_address
    mail['To'] = ', '.join(to_address)
    mail['Subject'] = Subject
    #將信件內文加到email中
    mail.attach(MIMEText(contents))       
    mail.attach(MIMEImage(image))  
    
    # 設定smtp伺服器並寄發信件    
    try:
        smtpserver = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        smtpserver.ehlo()
        smtpserver.login(gmail_user, gmail_password)
        smtpserver.sendmail(from_address, to_address, mail.as_string())
        smtpserver.quit()
        print("成功寄出")
    except Exception() as e:
        print(e)
    

#image = Path("1.jpg").read_bytes()
#SEND('allen01105@gmail.com',image)
            

