import os
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import numpy as np
import glob

### data loading ###

data=[]
target=[]

for i in os.listdir('./data/crescent moon'):
    path='./data/crescent moon/'+i
    img=cv2.imread(path)
    res=cv2.resize(img,dsize=(150,150),interpolation=cv2.INTER_CUBIC)
    data.append(res)
    target.append(0)

for i in os.listdir('./data/dark moon'):
    path='./data/dark moon/'+i
    img=cv2.imread(path)
    res=cv2.resize(img,dsize=(150,150),interpolation=cv2.INTER_CUBIC)
    data.append(res)
    target.append(1)

for i in os.listdir('./data/first quarter moon'):
    path='./data/first quarter moon/'+i
    img=cv2.imread(path)
    res=cv2.resize(img,dsize=(150,150),interpolation=cv2.INTER_CUBIC)
    data.append(res)
    target.append(2)

for i in os.listdir('./data/full moon'):
    path='./data/full moon/'+i
    img=cv2.imread(path)
    res=cv2.resize(img,dsize=(150,150),interpolation=cv2.INTER_CUBIC)
    data.append(res)
    target.append(3)

for i in os.listdir('./data/last quarter moon'):
    path='./data/last quarter moon/'+i
    img=cv2.imread(path)
    res=cv2.resize(img,dsize=(150,150),interpolation=cv2.INTER_CUBIC)
    data.append(res)
    target.append(4)

data_arr=np.array(data)
target_arr=np.array(target)

### 평균, 표준편차 plot ###

class_0_avg=[]
class_0_std=[]
class_1_avg=[]
class_1_std=[]
class_2_avg=[]
class_2_std=[]
class_3_avg=[]
class_3_std=[]
class_4_avg=[]
class_4_std=[]

path = '.data/last quarter moon' # 경로 입력 (클래스 바꿔가면서 반복)
exts = ['.jpg', '.jpeg'] # 확장자명 입력
data_list = []
for ext in exts:
	data_list+=glob.glob(path+'/*'+ext)
img_norm = list()
img_std = list()
img_var = list()
n=0
for data in data_list:
    img = cv2.imread(data, cv2.IMREAD_COLOR).astype(np.float32)
    if len(img.shape) <2: # 흑백 이미지는 제외
        continue
    mean,std = np.mean(img, axis=(0,1)), np.std(img, axis=(0,1))
    img_norm.append(mean)
    img_std.append(std)
    n+=1
total_img_norm=[]
total_img_std=[]
for i in range(len(img_norm)):
  total_img_norm.append((img_norm[i][0]+img_norm[i][1]+img_norm[i][2])/3.0)
for i in range(len(img_std)):
  total_img_std.append((img_std[i][0]+img_std[i][1]+img_std[i][2])/3.0)
class_4_avg=total_img_norm # 변수에 저장 (클래스 바꿔가면서 반복)
class_4_std=total_img_std # 변수에 저장 (클래스 바꿔가면서 반복)

plt.scatter(class_0_avg,class_0_std, color='r', label='First quarter moon', alpha=0.5)
plt.scatter(class_1_avg,class_1_std, color='b', label='Full moon', alpha=0.5)
plt.scatter(class_2_avg,class_2_std, color='g', label='Last quarter moon', alpha=0.5)
plt.scatter(class_3_avg,class_3_std, color='pink', label='Dark moon', alpha=0.5)
plt.scatter(class_4_avg,class_4_std, color='orange', label='Crescent moon', alpha=0.5)
plt.xlabel('Average value of pixel of image', fontsize=13)
plt.ylabel('Standard deviation', fontsize=13)
plt.legend()
plt.show()
plt.clf()




