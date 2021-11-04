import json

# 

right_file = r"D:\NLP_Bilibili\output鹿乃ちゃん_【b限】Apex【鹿乃まほろ】_1624111227443.json"
test_file = r"D:\NLP_Bilibili\BaiduAI\output_lunai (1).json"

with open(right_file, 'rb') as load_f:
    load_dict = json.load(load_f)

with open(test_file, 'rb') as load_ft:
    load_test_dict = json.load(load_ft)
    
acc = 0
pos_but_neg = 0
neg_but_pos = 0
bug = 0

for x, y in zip(load_dict, load_test_dict):
    if(x["label"] == y["label"]):
        acc += 1
    elif(x["label"] == "positive"):
        pos_but_neg += 1
    elif(x["label"] == "negative"):
        neg_but_pos += 1
    else:
        bug += 1

print("acc:", acc)
print(pos_but_neg)
print(neg_but_pos)
print("bug:", bug)