import json
import paddle
import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from utils import convert_example, create_dataloader
import paddle.nn.functional as F
from utils import evaluate
from utils import predict

# 使用paddlepaddle==2.0.1，注意环境配置


def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip('\n').split('\t')
            if len(l) != 2:
                print (len(l), line)
            words, label = line.strip('\n').split('\t')
            # words = words.split('\002')
            # label = labels.split('\002')
            yield {'text': words, 'label': label}

# data_path为read()方法的参数


train_ds = load_dataset(read, data_path='train.txt',lazy=False)
dev_ds = load_dataset(read, data_path='dev.txt',lazy=False)
train_ds.label_list = ['0', '1']
dev_ds.label_list = ['0', '1']

print(train_ds.label_list)

for data in train_ds.data[:5]:
    print(data)

# 设置想要使用模型的名称
MODEL_NAME = "ernie-1.0"

tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
ernie_model = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)

# 模型运行批处理大小
batch_size = 1  # 本机GPU只能装下1个batch，在百度GPU上可以设为32
max_seq_length = 128

# dataloader
# python中的偏函数partial，把一个函数的某些参数固定住（也就是设置默认值），
# 返回一个新的函数，调用这个新函数会更简单。
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)

# 将读入的数据batch化处理，便于模型batch化运算。
# batch中的每个句子将会padding到这个batch中的文本最大长度batch_max_seq_len。
# 当文本长度大于batch_max_seq时，将会截断到batch_max_seq_len；当文本长度小于batch_max_seq时，将会padding补齐到batch_max_seq_len.

batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]

train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)

ernie_model = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)

model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=len(train_ds.label_list))

# 训练过程中的最大学习率
learning_rate = 5e-5
# 训练轮次
epochs = 1  # 3
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01

num_training_steps = len(train_data_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

global_step = 0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
    evaluate(model, criterion, metric, dev_data_loader)

model.save_pretrained('./checkpoint')
tokenizer.save_pretrained('./checkpoint')


# 预测工作也放在这个脚本里了
# 输入需要经由 处理，输出到文件output_file中

input_file = "nanami.json"
output_file = "output_nanami.json"
with open(input_file, 'rb') as load_f:
    load_dict = json.load(load_f)

# data = [
#     {"text":'这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般'},
#     {"text":'怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片'},
#     {"text":'作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。'},
# ]
data = load_dict
label_map = {0: 'negative', 1: 'positive'}

results = predict(
    model, data, tokenizer, label_map, batch_size=batch_size)

senta_list = []
tt = {}
for idx, text in enumerate(data):
    # print('Data: {} \t Lable: {}'.format(text, results[idx]))
    tt["text"] = text["text"]
    tt["label"] = results[idx]
    senta_list.append(tt)
    tt = {}


with open(output_file, "w", encoding="utf-8") as dump_f:
    json.dump(senta_list, dump_f, ensure_ascii=False)
