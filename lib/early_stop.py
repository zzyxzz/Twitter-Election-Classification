class EarlyStopping(object):
    def __init__(self, threshold=5):
        self.threshod = threshold
        self.scores = []
        # performance decrease
        self.consecutive_decrease = 0

    def stop_training(self, score, step):
        '''
        :param score: score on validation set
        :param step: iteration step
        :return: boolean
        '''
        self.scores.append(score)

        if len(self.scores) > 1 and self.scores[-1] > self.scores[-2]:
            self.consecutive_decrease += 1
        else:
            self.consecutive_decrease = 0
        # print self.consecutive_decrease, self.scores
        if self.consecutive_decrease >= self.threshod:
            print "Early stopping criteria {} reached! Stopped at step: {}".format(self.threshod, step)
            return True
        else:
            return False
