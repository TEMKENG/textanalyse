import sys
from utils import *
import pandas as pd
from icecream import ic
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from simpletransformers.classification import ClassificationModel


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')


train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "num_train_epochs": 2,
    "train_batch_size": 4,
    "fp16": False,
}
# define hyperparameter

# Create a ClassificationModel
model = ClassificationModel(
    "bert", "distilbert-base-german-cased",
    num_labels=4,
    use_cuda=True,
    args=train_args
)
ic(sys.getsizeof(model))
DATASET_DIR = "../datasets/germeval"
class_list = ['INSULT', 'ABUSE', 'PROFANITY', 'OTHER']

df1 = pd.read_csv(join(DATASET_DIR, 'germeval2019GoldLabelsSubtask1_2.txt'), sep='\t', lineterminator='\n',
                  encoding='utf8', names=["tweet", "task1", "task2"])
df2 = pd.read_csv(join(DATASET_DIR, 'germeval2019.training_subtask1_2_korrigiert.txt'), sep='\t', lineterminator='\n',
                  encoding='utf8', names=["tweet", "task1", "task2"])

df = pd.concat([df1, df2])
df['task2'] = df['task2'].str.replace('\r', "")
df['pred_class'] = df.apply(lambda x: class_list.index(x['task2']), axis=1)

df = df[['tweet', 'pred_class']]

print(df.shape)
print(df.head())
# Daten aufteilen
train_df, test_df = train_test_split(df, test_size=0.10)

print('train shape: ', train_df.shape)
print('test shape: ', test_df.shape)

# train shape:  (6309, 2)
# test shape:  (702, 2)
# model.train_model(train_df)
model.train_model(test_df)

result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=f1_multiclass, acc=accuracy_score)
ic(result)
ic(model_outputs)
# ic(wrong_predictions)
# {'acc': 0.6894586894586895,
# 'eval_loss': 0.8673831869594075,
# 'f1': 0.6894586894586895,
# 'mcc': 0.25262380289641617}
