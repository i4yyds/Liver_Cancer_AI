# -*- coding: utf-8 -*-
import cv2
from multiprocessing import Process
import os
import numpy as np
import pydicom as dcm
import math
import scipy.stats

class dataProcess():
    def __init__(self, data_dir_in,data_dir_out, sample_num=20,process_num=30):
        self.data_dir_in = data_dir_in  
        self.data_dir_out = data_dir_out
        self.process_num = process_num
        self.sample_num = sample_num
        self.imagelist = os.listdir(data_dir_in)
        self.image_num = len(self.imagelist)
        self.numlist = self.count_process_num()
        self.check_dir()
        
        
    def check_dir(self):
        if os.path.exists(self.data_dir_out) == False:
            os.mkdir(self.data_dir_out)
    
    def count_process_num(self):
        num = np.zeros(self.process_num,dtype=int)
        out = np.zeros((self.process_num,2),dtype=int)
        for i in range(self.process_num):
            if i == self.process_num-1:
                num[i] = int(self.image_num / self.process_num + self.image_num % self.process_num)
            else:
                num[i] = int(self.image_num / self.process_num)
        #print num
        for i in range(self.process_num):
            if i == 0:
                out[i][0] = 0
                out[i][1] = num[i]
            else:
                out[i][0] = out[i-1][1]
                out[i][1] = out[i-1][1] + num[i]
        return out
        
    def remove_point(self,image,m,n):
        mean = (int(image[m-1][n-1]) + int(image[m-1][n]) + int(image[m-1][n+1])
            + int(image[m][n-1])    +         0          + int(image[m][n+1])
            + int(image[m+1][n-1])  + int(image[m+1][n]) + int(image[m+1][n+1]))/8
        if abs(image[m][n]-mean) > 20:
            return mean
        else:
            return image[m][n]
        
    def opendcmfile(self,file_name):
        images = dcm.read_file(file_name)
        images.image = images.pixel_array * images.RescaleSlope + images.RescaleIntercept
        slices = []
        slices.append(images)
        image = slices[int(len(slices) / 2)].image.copy()
        tmpimage = image.reshape((512, 512, 1))
        return np.array(tmpimage)
        
    def findname(self,dir_name,file_name):
        for i in [x for x in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name,x)) and file_name in x]:
            return i#os.path.join(dir_name,i)
        
    def getimgListAndNum(self,dir_name):
        file_list = []
        length = len(os.listdir(dir_name))
        add = 0
        # for i in range(length):
        #     file_name = self.findname(dir_name,str(i+add).zfill(4)+'.dcm')
        #     if file_name == None:
        #         add = add + 1
        #         file_name = self.findname(dir_name,str(i+add).zfill(4)+'.dcm')
        #     file_list.append(file_name)
        for i in range(length):
            file_name = self.findname(dir_name,str(i+1).zfill(4)+'.dcm')
            if file_name == None:
                file_name = self.findname(dir_name, 'image-' + str(i+1)+'.dcm')
            file_list.append(file_name)
        
        return file_list,length
        
    def getNDIndexList(self,IndexNum, TotalNum, u=0, a=1):
        IndexList = []
        boundary = 2
        
        if IndexNum >= TotalNum:
            sub = IndexNum - TotalNum
            mid = math.floor(TotalNum / 2)
            up = math.floor(sub / 2)
            start = int(mid - up)
            end = int(start + sub)
            
            for i in range(TotalNum):
                IndexList.append(i)
                if i in range(start,end):
                    IndexList.append(i)
            assert len(IndexList) == IndexNum
                    
        elif IndexNum < TotalNum:
            mid = math.floor(IndexNum / 2)
            step = boundary / mid
            index = 0
            P_sum = 0.0
            P = []
            for i in range(int(mid)):
                index = index + step
                probability = scipy.stats.norm(u,a).cdf(index) - scipy.stats.norm(u,a).cdf(index-step)
                P.append(probability)
                P_sum = P_sum + probability
            P = P / P_sum
            
            Out_P = []
            
            for i in range(len(P)):
                Out = 0
                for j in range(i+1):
                    Out = Out + P[j]
                Out_P.append(Out)
            
            Total_mid = math.floor(TotalNum / 2)
            for i in range(int(mid)):
                IndexList.append(int(round(Total_mid * Out_P[i])))
            
            for i in range(int(mid)-1,-1,-1):
                IndexList.append(TotalNum-int(round(Total_mid * Out_P[i])))
    
        assert len(IndexList) == IndexNum
        return IndexList
        
    def getImageSet(self,dir_name):
        img_List, img_Num = self.getimgListAndNum(dir_name)
        IndexList = self.getNDIndexList(self.sample_num, img_Num)
        
        imageset = []
        for i in range(self.sample_num):
            img = self.opendcmfile(dir_name + img_List[IndexList[i]])
            imageset.append(img)
        
        return np.array(imageset)
        
    def ImageProcess(self, j, start, end):
        print('The %d Process Start!!' % j)
        for k in range(start,end):
            print('Process %d: %f%%' % (j,(k-start)/float(end-start)*100))
            if os.path.exists(self.data_dir_out+self.imagelist[k])==True:
                continue
            else:
                os.mkdir(self.data_dir_out+self.imagelist[k])
            
            image_path = self.data_dir_in + self.imagelist[k] + '/'
            imageset = self.getImageSet(image_path)
            
            for i in range(self.sample_num):
                image = imageset[i]
                '''dim =image.shape
                for m in range(2,dim[0]-2):
                    for n in range(2,dim[1]-2):
                        image[m][n] = self.remove_point(image,m,n)'''
                cv2.imwrite(self.data_dir_out+self.imagelist[k]+"/"+str(i+1)+".bmp",image)
    
    def run(self):
        subProcessList = []
        for i in range(self.process_num):
            p = Process(target=self.ImageProcess, args=(i,data.numlist[i][0],data.numlist[i][1],))
            p.daemon = True
            p.start()
            subProcessList.append(p)
        for p in subProcessList:
            p.join()
        print('All subprocesses done.')
        
if __name__=='__main__':
    data = dataProcess('a/',
                       'a_process/')
    data.run()