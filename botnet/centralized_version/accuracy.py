import numpy as np
from predict import predict

def accuracy (w, b, np_Xy):
    correctly_classified = 0;
    for element in np_Xy:
        if(predict(w,b,element[0:11]) == element[11]):
            correctly_classified += 1

    accuracy = correctly_classified/np_Xy.shape[0]
    return accuracy