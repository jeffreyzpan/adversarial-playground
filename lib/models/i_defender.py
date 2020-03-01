import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

class I_Defender():
    def __init__(self, classifier, dataloader, num_classes, p_value, mode='GMM', n_components=10, max_iter=100, n_init=1, gpu_ids='0,1,2,3', **kwargs):
        self.classifier = classifier
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.p_value = p_value
        self.gpu_ids = gpu_ids

        if mode =='GMM':
                self.model_list = [GaussianMixture(n_components=n_components, max_iter=max_iter, n_init=n_init) for i in range(num_classes)]
        elif mode =='autoencoder':
            raise NotImplementedError("not yet implemented") #TODO implement autoencoder arch
        else:
            raise NotImplementedError

        self.capture_hidden_states()
        print('fitting GMM model')
        for i in range(num_classes):
            self.model_list[i].fit(self.IHSD_list[i])

    def capture_hidden_states(self):
        self.IHSD_list = [[] for i in range(self.num_classes)]
        with torch.no_grad():

            for i, (images, target) in enumerate(self.dataloader):
                if '-1' not in self.gpu_ids:
                    images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # pass input through model and record the hidden state
                output = self.classifier(images)
                # get seen hidden state of shape [linear_layer, num batches, 1, linear layer size]
                hidden_state = self.classifier.module.return_hidden_state_memory()
                # only use the last layer for fitting GMMs
                for label, state in zip(target, torch.cat(hidden_state[-1])):
                    self.IHSD_list[label].append(state.data.cpu().numpy())
        self.IHSD_list = [np.array(i) for i in self.IHSD_list]
        
    def estimate(self, x, predicted_label):
        log_prob = self.model_list[predicted_label].score_samples(x.data.cpu().numpy())
        return log_prob < self.p_value, log_prob

def i_defender(classifier, dataloader, num_classes, p_value, mode='GMM', n_components=10, max_iter=500, n_init=5, gpu_ids='0,1,2,3'):
    model = I_Defender(classifier, dataloader, num_classes, p_value, mode, n_components, max_iter, n_init, gpu_ids)
    return model
