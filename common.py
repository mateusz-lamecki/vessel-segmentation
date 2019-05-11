from sklearn import metrics

DATA_PATH = 'data/'
X_IMG_PATH = DATA_PATH + 'images/'
Y_IMG_PATH = DATA_PATH + 'labels/'


class Metrics:
    def __init__(self):
        self.acc = 0
        self.sensitivity = 0
        self.specificity = 0
        self.f1 = 0
        
    def calculate(self, y_true, y_pred):
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        
        self.acc = metrics.accuracy_score(y_true, y_pred)
        self.sensitivity = tp = tp / (tp + fn)
        self.specificity = tn / (tn + fp)
        self.f1 = metrics.f1_score(y_true, y_pred)
        
    def __str__(self):
        return 'Acc: {:.5f}, sensitivity: {:.5f}, specificity: {:.5f}, F1-score: {:.5f}'.format(
            self.acc, self.sensitivity, self.specificity, self.f1)
    
    @staticmethod
    def mean_metric(scores_all):
        """ Calculates mean metrics for samples """
        res = Metrics()
        
        for metric in scores_all:
            res.acc += metric.acc / len(scores_all)
            res.sensitivity += metric.sensitivity / len(scores_all)
            res.specificity += metric.specificity / len(scores_all)
            res.f1 += metric.f1 / len(scores_all)
            
        return res


def get_filenames():
    filenames = os.listdir(X_IMG_PATH)
    return filenames
