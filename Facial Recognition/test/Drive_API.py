# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:25:42 2020

@author: allen
"""
from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
import cv2
# In[全域變數]
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)
    service = build('drive', 'v3', credentials=creds)

# In[]
def Drive_upload(path,name):
    global service
    
    if name == None:
        return 0 
    
    #-------------上傳雲端-------------------

    results = service.files().list(
        pageSize=10, fields="nextPageToken, files(id, name)").execute()
    
    items = results.get('files', [])
    
    #如果這個名字已經有資料夾，把id記起來
    folder_id = None
    for item in items:
        #print(item['name'])
        if item['name'] == name:
            folder_id = item['id']
            print("找到 "+ name + "資料夾")
            
    #如果沒有該使用者的資料，創一個資料夾，並把id記起來
    if folder_id == None:
        folder_id = Drive_create_folder(name)
    
        
    file_metadata = {'name': name + ".jpg",
                     'parents': [folder_id]}
        
    media = MediaFileUpload(path,
                            mimetype='image/jpeg')
        
    file = service.files().create(body=file_metadata,
                                  media_body=media,
                                  fields='id').execute()
        
    #print('File ID: %s' % file.get('id'))
    print("圖片上傳雲端完成")

# In[]
def Drive_create_folder(folder_name):
    global service
    
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    
    file =service.files().create(body=file_metadata,
                                 fields='id').execute()
    print("創建" + folder_name + "資料夾完成")
    #print('Folder ID: %s' % file.get('id'))
    return file.get('id')

# In[1]
if __name__ == '__main__':
    Drive_upload(None, None)