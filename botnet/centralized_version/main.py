from preprocess import readFile
from train import train

if __name__ == "__main__":
    # read data
    data = readFile("data/botnet_tot_syn_l.csv")
    # standardize
    # train
    weights, bias = train(data, 10, 1.5, 0.05)

