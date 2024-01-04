def predict_single(customer, dv, model):
    x = dv.transform([customer])
    y_pred = model.predict_proba(x)[:, 1]
    return float(y_pred[0])
