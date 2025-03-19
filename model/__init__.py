from model.Optimizers import build_Adam
from model.loss_function import build_CrossentropyLoss_ContrastiveLoss, build_BCELoss, build_CrossEntropyLoss, build_CrossEntropyLoss_weighted
import model.TRAR
import model.attention
from model.FED import build_FED
from model.attention.GuideAttentionLayer import GuideAttentionLayer
from model.attention.TraditionalAttentionLayer import TraditionalAttentionLayer
_models={
    "FED":build_FED
}

_optimizers={
    "Adam":build_Adam
}

_loss={
    "CrossEntropyLoss":build_CrossEntropyLoss,
    "BCELoss":build_BCELoss,
    "CrossentropyLoss_ContrastiveLoss": build_CrossentropyLoss_ContrastiveLoss,
    "Crossentropy_Loss_weighted": build_CrossEntropyLoss_weighted
}