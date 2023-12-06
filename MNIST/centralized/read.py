import pandas as pd
def serialReadFile(filename):
    data = pd.read_csv(filename)

    data.drop("label", axis=1, inplace=True)

    return data
