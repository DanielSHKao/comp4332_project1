import torch
import torch.nn.functional as F
import glob
from argparse import ArgumentParser
from utils.utils import *
from torchmetrics import Accuracy
class Ensembler:
    def __init__(self,models,hparams):
        self.models = models
        self.hparams = hparams
    def predict(self,x_batch):
        results = torch.zeros((x_batch.size(0), self.hparams.num_classes))
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x_batch)
            pred = torch.argmax(pred,dim=1)
            pred = F.one_hot(pred, num_classes=self.hparams.num_classes)
            results += pred
        return results

def pred_valid_stars(hparams, metric):
    if hparams.test_models=='all':
        models_path = glob.glob("./pretrain/*.pt")
    else:
        models_path = hparams.test_models
    sent_models=list()
    word_models=list()
    for model_path in models_path:
        if 'mlp' in model_path or 'dnn' in model_path:
            sent_models.append(model_path)
        else:
            word_models.append(model_path)
    sent_models = [torch.load(m) for m in sent_models]
    word_models = [torch.load(m) for m in word_models]
    sent_ensembler = Ensembler(sent_models,hparams)
    word_ensembler = Ensembler(word_models,hparams)
    try:
        sent_df = load_preprocess('valid',columns=['text','stars'],folder=hparams.data_dir,embedding='sentence')
    except:
        sent_df = None
    try:
        word_df = load_preprocess('valid',columns=['text','stars'],folder=hparams.data_dir,embedding='word')
    except:
        word_df = None
    if sent_df is None and word_df is None:
        print("No embedded dataset founded. (Train once to embed the data.)" )
        return
    gt = torch.tensor(sent_df['stars'].tolist())-1
    result = torch.zeros((len(gt),5))
    if sent_df is not None:
        sent_vector = torch.tensor(sent_df['sent_embed'].tolist())
        result += sent_ensembler.predict(sent_vector)
    if word_df is not None:
        word_vector = torch.tensor(word_df['word_embed'].tolist())
        result +=word_ensembler.predict(word_vector)
    result = torch.argmax(result,dim=1)
    return metric(result,gt), models_path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/proj1.yaml')
    parser.add_argument("--data_dir", type=str, default="data/")
    
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)

    print('Testing ensemble validation dataset...')
    metric = Accuracy(task="multiclass", num_classes=5)
    result, models = pred_valid_stars(args,metric)
    print("--------------------------------------------------------")
    print(f"model found: {models}")
    print(f"Validation accuracy: {result}")



