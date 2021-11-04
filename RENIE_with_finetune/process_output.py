import json

# 从BiLSTM模型训练结束后的输出文件 生成 预测时的输入文件

output_file = r"D:\NLP_Bilibili\output_5s_BILSTM_文静看看二创闲聊__1633529522047.json"
new_file = r"D:\NLP_Bilibili\文静闲聊.json"

with open(output_file, 'rb') as load_f:
    load_dict = json.load(load_f)

senta_list = []
tem = {}
for item in load_dict:
    tem["text"] = item["text"]
    senta_list.append(tem)
    tem = {}


with open(new_file, "w", encoding="utf-8") as dump_f:
    json.dump(senta_list, dump_f, ensure_ascii=False)

