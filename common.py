import matplotlib.pyplot as plt
from sklearn import metrics
import cv2


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


def load_img(path):
    """ Loads and returns image in RGB color model """
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def plot_vessel(images_orig, y_trues, y_preds, save_file=None):
    fig = plt.figure(figsize=(10, 50))
    cols, rows = 3, len(images_orig)

    for i in range(rows):
        fig.add_subplot(rows, cols, cols*i + 1)
        plt.imshow(images_orig[i], aspect='equal')

        fig.add_subplot(rows, cols, cols*i + 2)
        plt.imshow(y_trues[i], aspect='equal')

        fig.add_subplot(rows, cols, cols*i + 3)
        y_pred_mask = __apply_mask_with_color(images_orig[i], y_preds[i])
        plt.imshow(y_pred_mask, aspect='equal')

        if save_file is not None:
            plt.savefig(save_file)

def __apply_mask_with_color(img, mask):
    """ Applies colored mask on original image """
    img = img.copy()
    img[mask] = (60, 255, 0)
    return img

