from accuracy import accuracy
from normalize import normalize
from preprocess import readFile
from train import train

if __name__ == "__main__":
    # read data
    data = readFile("botnet/data/botnet_tot_syn_l.csv")
    # standardize
    data = normalize(data)
    # train
    weights, bias = train(data, 10, 1.5, 0.05)
    # accuracy
    accuracy = accuracy(weights, bias, data)
    print("accuracy:", accuracy)
