from app import predict_top_5_dl

def test_prediction():
    question = "Comment créer une API avec Python ?"
    result = predict_top_5_dl(question)

    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][0], str)
    assert isinstance(result[0][1], float)
