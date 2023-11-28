from preprocess import readFile
from train import train
from normalize import normalize
from accuracy import accuracy

if __name__ == "__main__":
    # read data
    data = readFile("data/botnet_tot_syn_l.csv")
    # standardize
    data = normalize(data)
    # train
    weights, bias = train(data, 10, 1.5, 0.05)
    # accuracy
    accuracy = accuracy(weights, bias, data)
    print("accuracy:", accuracy)
