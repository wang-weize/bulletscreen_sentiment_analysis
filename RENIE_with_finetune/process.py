import json

# 从BiLSTM模型训练结束后的输出文件 生成 训练时的输入文件

output_file = r"D:\NLP_Bilibili\output鹿乃ちゃん_【b限】Apex【鹿乃まほろ】_1624111227443.json"
txt_file = r"D:\NLP_Bilibili\lunaiall.txt"

with open(output_file, 'rb') as load_f:
    load_dict = json.load(load_f)

bad = 0
good = 0
all = 0
with open(txt_file, "w", encoding="utf-8") as dump_f:
    for item in load_dict:
        dump_f.write(item["text"])
        dump_f.write("\t")
        if item["label"] == "positive":
            dump_f.write("1")
            good += 1
        else:
            dump_f.write("0")
            bad += 1
        dump_f.write("\n")

print(bad)
all = good + bad
print(all)