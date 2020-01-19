import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

class I_Defender():
    def __init__(self, classifier, dataloader, benchmark, mode='GMM', device='gpu', **kwargs):
        self.classifer = classifier
        self.dataloader = dataloader
        self.benchmark = benchmark
        if mode =='GMM':
            self.model = GaussianMixture()
        elif mode =='autoencoder':
            raise NotImplementedError, "not yet implemented" #TODO implement autoencoder arch
        else:
            raise NotImplementedError

        self.activation_states = None
    
    def capture_hidden_states(self):
        with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(self.dataloader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # pass input through model and record the hidden state
            output = model(images)
        
        self.IHSD = model.return_hidden_state_memory(self)
        
