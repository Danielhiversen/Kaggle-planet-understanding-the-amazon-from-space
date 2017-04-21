# https://www.kaggle.com/paulorzp/planet-understanding-the-amazon-from-space/f2-score/run/1096439

from sklearn.metrics import fbeta_score

labels = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
          'blow_down', 'clear', 'cloudy', 'conventional_mine', 'cultivation',
          'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 
          'selective_logging', 'slash_burn', 'water']

def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, labels=labels, beta=2, average='macro')
