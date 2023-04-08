from sklearn.metrics import f1_score

def MicroF1(outputs,labels):
    return f1_score(outputs.cpu(), labels.cpu(), average='micro')

def MacroF1(outputs,labels):
    return f1_score(outputs.cpu(), labels.cpu(), average='macro')