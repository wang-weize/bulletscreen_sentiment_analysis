import matplotlib.pyplot as plt
import numpy as np
import json

from matplotlib import ticker
# 此程序运行于pycharm，可以将弹幕情感、弹幕密度打印成柱状图
# 横坐标是时间，单位为time_tap秒；纵坐标的正负由情感的正负决定，大小=时间段弹幕数
# 图上右键“重构-复制文件”保存图片

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (100, 10)

origin_file = "文静_千鸟Official_【突击】看看二创闲聊__1633529522047.json"
output_file = "output_文静闲聊重训练.json"
time_tap = 5

with open(origin_file, 'rb') as load_f:
    load_dict = json.load(load_f)

comment_list = []
init_time = int(load_dict["full_comments"][0]["time"] / 1000)
tmpstr = 0
comment_frequency_list = []

for item in load_dict["full_comments"]:
    try:
        comment_list.append(item["text"])
        # print(item['text'])
        new_time = int(item["time"] / 1000)
        if new_time < init_time + time_tap:
            tmpstr += 1
        else:
            comment_frequency_list.append(tmpstr)
            init_time += time_tap
            tmpstr = 0
    except:
        pass

with open(output_file, 'rb') as load_f:
    load_dict = json.load(load_f)

label_list = []
color_list = []
i = 0
neg = 0
all_comment = 0

for item in load_dict:
    if item["label"] == "negative":
        label_list.append(-comment_frequency_list[i])
        color_list.append("black")
        neg += 1
    else:
        label_list.append(comment_frequency_list[i])
        color_list.append("red")
    i += 1
    all_comment += 1

time_list = []
tim = 0
for item in load_dict:
    m, s = divmod(tim, 60)
    h, m = divmod(m, 60)
    time_list.append("%d:%02d:%02d" % (h, m, s))
    tim += time_tap

x = np.array(time_list)
y = np.array(label_list)
print(y)
print(neg)
print(all_comment)
print(neg / all_comment)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(30))
plt.bar(x, y, width=1, color=color_list)
plt.show()