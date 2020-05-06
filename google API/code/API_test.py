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

def main():
    case = input("輸入數字來選擇接下來的操作\n1 : 檢視資料夾\n2 : 新增資料夾\n3 : 上傳檔案\n")
    print("case = ",case)
    Case_Detect(case)
    
def Case_Detect(case):
    
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
        service = build('drive', 'v3', credentials=creds)
        
    if case== '1' :
        # Call the Drive v3 API
        results = service.files().list(
            pageSize=10, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])
    
        if not items:
            print('No files found.')
        else:
            print('Files:')
            for item in items:
                print(u'{0} ({1})'.format(item['name'], item['id']))
    if case== '2' :
        name = input("輸入資料夾名稱:\n")
        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        file =service.files().create(body=file_metadata,
                                            fields='id').execute()
        print('Folder ID: %s' % file.get('id'))
        
    if case == '3':
        results = service.files().list(
            pageSize=10, fields="nextPageToken, files(id, name)").execute()
        
        items = results.get('files', [])
        
        name = input("目的地資料夾名稱:\n")
        
        
        
        folder_id = None
        for item in items:
            if item['name'] == name :
                folder_id = item['id']
                
        if folder_id == None:
            print("無效資料夾名稱")
            return 0 
        
        #print(folder_id)
            
        path = os.listdir('../picture')
        #print(path)
        
        for name in path:
            
            file_metadata = {'name': name,
                             'parents': [folder_id]}
            
            media = MediaFileUpload('../picture/'+ name,
                                    mimetype='image/jpeg')
            
            file = service.files().create(body=file_metadata,
                                                media_body=media,
                                                fields='id').execute()
            
            print('File ID: %s' % file.get('id'))
        
    
if __name__ == '__main__':
    main()