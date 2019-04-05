import scipy.io
from datetime import datetime, timedelta
mat = scipy.io.loadmat('wiki.mat')['wiki'][0][0]
dob = [datetime.fromordinal(int(date)) - timedelta(days = 366) for date in mat[0][0]]
photo_taken = mat[1][0]
full_path = mat[2][0]
gender = mat[3][0]
name = mat[4][0]
face_location = mat[5][0]
face_score = mat[6][0]
second_face_score = mat[7][0]


# 筛选/00文件夹中的600张图片
age=[]
filted_index = [index for index,path in enumerate(full_path) if path[0][:2] == '00' ]
for index in filted_index:
    age.append({'filename': full_path[index][0][3:], 'age':photo_taken[index] - dob[index].year + dob[index].month/12.0})
#排除负的数据
age = [age_e for age_e in age if age_e['age']>0 and age_e['age']<=100]
a=0

# 筛选/01文件夹中的600张图片
age_test = []
filted_index = [index for index,path in enumerate(full_path) if path[0][:2] == '01' ]
for index in filted_index:
    age_test.append({'filename': full_path[index][0][3:], 'age':photo_taken[index] - dob[index].year + dob[index].month/12.0})
#排除负的数据
age_test = [age_e for age_e in age if age_e['age']>0 and age_e['age']<=100]
a=0
