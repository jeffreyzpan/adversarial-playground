import torch
import numpy

def gen_attacks(test_images, test_labels, classifier, criterion, attacks): 

    adv_dict = {}

    # loop through list of attacks and generate adversarial images using the given method
    for attack_name, attack in zip(attacks.keys(), attacks.values()):

        #hopskipjump requires multiple iterations for good results
        if attack_name == 'hopskipjump':
            adv_test = None
            for i in range(3):
                adv_test = attack.generate(x=test_images, x_adv_init=x_adv, resume=True)
        else:
            adv_test = attack.generate(x=test_images)

        #convert np array of adv. images to PyTorch dataloader for CUDA validation later
        adv_tensor = torch.Tensor(adv_test)
        adv_set = torch.utils.data.TensorDataset(adv_tensor, torch.Tensor(test_labels).long())
        adv_loader = torch.utils.data.DataLoader(adv_set)
        adv_dict[attack_name] = adv_loader

    return adv_dict

def gen_defences(test_images, adv_images, attack_name, test_labels, classifier, criterion, defences):
    
    def_adv_dict = {}

    # loop through list of defenses and generate defended images using the given method if method isn't adv. training based
    for defence_name, defence in zip(defences.keys(), defences.values()):

        #ART defences take in w x h x c, while original input is (c, w, h)
        #adv_images = np.moveaxis(adv_images, 1, -1)
        def_adv, _ = defence(adv_images)

        #switch channel axis for conversion back to PyTorch
        #def_adv = np.moveaxis(def_adv, -1, 1)

        #convert np array of defended images to PyTorch dataloader for CUDA validation later

        def_adv_set = torch.utils.data.TensorDataset(torch.Tensor(def_adv), torch.Tensor(test_labels).long())
        def_adv_loader = torch.utils.data.DataLoader(def_adv_set)
        def_adv_dict[defence_name] = def_adv_loader

    return def_adv_dict
