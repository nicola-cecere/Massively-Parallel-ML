from preprocess import readFile

if __name__ == "__main__":
    # read data
    data = readFile("data/botnet_tot_syn_l.csv")
    print(data)
    # standardize
