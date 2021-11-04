import json
from paddlenlp import Taskflow


def print_hi(name):
    print(f'Hi, {name}')  


# 输入文件是网站上的json文件，输出文件是[{"text":"", "label":"positive"},{},...]格式
# 每隔time_tap秒，把此时间段内的弹幕拼成一个字符串，作为BiLSTM的输入
origin_file = "文静_千鸟Official_【突击】看看二创闲聊__1633529522047.json"
output_file = "output_文静闲聊重训练.json"
time_tap = 5


if __name__ == '__main__':
    print_hi('PyCharm')
    with open(origin_file, 'rb') as load_f:
        load_dict = json.load(load_f)
        # print(load_dict)

    comment_list = []
    init_time = int(load_dict["full_comments"][0]["time"] / 1000)
    tmpstr = ""
    comment_time_list = []

    for item in load_dict["full_comments"]:
        try:
            comment_list.append(item["text"])
            print(item['text'])
            new_time = int(item["time"] / 1000)
            if new_time < init_time + time_tap:
                tmpstr += item["text"]
            else:
                comment_time_list.append(tmpstr)
                init_time += time_tap
                tmpstr = ""

        except:
            pass

    print(comment_time_list)
    senta = Taskflow("sentiment_analysis")
    senta_list = senta(comment_time_list)
    print(senta_list)

    all_sum = 0
    neg_sum = 0
    pos_sum = 0
    for ob in senta_list:
        if ob['label'] == 'negative':
            neg_sum += 1
        elif ob['label'] == 'positive':
            pos_sum += 1
        all_sum += 1

    neg_per = neg_sum / all_sum
    pos_per = neg_per / all_sum
    print("neg: ")
    print(neg_sum, neg_per)

    with open(output_file, "w", encoding="utf-8") as dump_f:
        json.dump(senta_list, dump_f, ensure_ascii=False)

