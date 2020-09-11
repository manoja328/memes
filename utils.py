from sklearn.metrics import roc_curve, roc_auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def rocCurve(actual, guesses, ax):
    fpr, tpr, _ = roc_curve(actual, guesses)
    roc_auc = roc_auc_score(actual, guesses)
    ax.step(fpr, tpr, color='purple', alpha=0.8, where='post', label='Area Under Curve: %1.4f' % roc_auc)
    ax.fill_between(fpr, tpr, step='post', alpha=0.8, color='purple')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_ylim([0., 1.])
    ax.set_xlim([0., 1.])
    ax.set_title('ROC Curve')
    ax.legend()
    return ax, roc_auc
