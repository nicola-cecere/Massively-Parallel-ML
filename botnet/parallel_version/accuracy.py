from predict import predict

from predict import predict

def calculate_accuracy(w, b, RDD_xy):
    prediction_results = RDD_xy.map(lambda record: 1 if predict(w, b, record[0]) == record[1] else 0)

    # Step 2: Use reduce to sum up the correct predictions
    correctly_classified = prediction_results.reduce(lambda a, b: a + b)

    # Step 4: Calculate accuracy
    accuracy = correctly_classified / RDD_xy.count()
    return accuracy



