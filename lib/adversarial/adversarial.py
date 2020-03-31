import torch
import numpy as np
import art.defences as defences

def gen_attacks(test_loader, classifier, attacks, epsilons, use_gpu=True): 

    adv_dict = {}
    test_labels = torch.tensor(test_loader.dataset.targets).long()

    # loop through list of attacks and generate adversarial images using the given method
    for attack_name, attack in zip(attacks.keys(), attacks.values()):

        print(attack_name)
        adv_list = [[] for i in range(len(epsilons))]
        for i, (inputs, target) in enumerate(test_loader):
            print(i)
            if use_gpu:
                inputs = inputs.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            _, adv, success = attack(classifier, inputs, target, epsilons=epsilons)
            for i, adv_images in enumerate(adv):
                adv_list[i].append(adv_images.cpu()) 
            robust_accuracy = 1.0 - success.cpu().numpy().mean(axis=-1)

        for i, adv_examples in enumerate(adv_list):
            adv_examples = torch.cat(adv_examples, axis=0)
             #convert list of adversarial images to PyTorch dataloader for CUDA validation later
            adv_set = torch.utils.data.TensorDataset(adv_examples, test_labels)
            adv_loader = torch.utils.data.DataLoader(adv_set, batch_size=128, num_workers=16)
            adv_list[i] = adv_loader
        adv_dict[attack_name] = adv_list
        print('done')

    return adv_dict

def gen_defences(test_images, adv_images, attack_name, test_labels, defences):
    
    def_adv_dict = {}

    # loop through list of defenses and generate defended images using the given method if method isn't adv. training based
    for defence_name, defence in zip(defences.keys(), defences.values()):
        
        print(defence_name)

        #ART defences take in w x h x c, while original input is (c, w, h)
        #adv_images = np.moveaxis(adv_images, 1, -1)
        
        def_adv, _ = defence(adv_images)

        #switch channel axis for conversion back to PyTorch
        #def_adv = np.moveaxis(def_adv, -1, 1)

        #convert np array of defended images to PyTorch dataloader for CUDA validation later

        def_adv_set = torch.utils.data.TensorDataset(torch.from_numpy(def_adv), torch.from_numpy(test_labels).long())
        def_adv_loader = torch.utils.data.DataLoader(def_adv_set, batch_size=128, num_workers=16)
        def_adv_dict[defence_name] = def_adv_loader

    return def_adv_dict

def adversarial_retraining(clean_dataloader, attack):

    # attack training set images with given attack and save it to a new dataloader
    # ART appears to only support numpy arrays, so convert dataloader into a numpy array of images
    clean_image_batches, clean_label_batches = zip(*[batch for batch in clean_dataloader])
    clean_images = torch.cat(clean_image_batches).numpy()
    clean_labels = torch.cat(clean_label_batches).numpy() 

    adv_images = attack.generate(x=clean_images)
    adv_set = torch.utils.data.TensorDataset(torch.from_numpy(adv_images), torch.from_numpy(clean_labels).long())
    adv_loader = torch.utils.data.DataLoader(adv_set, batch_size=128, num_workers=16)
    
    return adv_loader 

def thermometer_encoding(train_loader, adv_loader, thm_params, save=False):

    encoding = defences.ThermometerEncoding(clip_values=(thm_params['clip_min'], thm_params['clip_max']), num_space=thm_params['num_space'], channel_index=thm_params['channel_index']) 
    
    print('Generating thermometer encoded images') 
    clean_image_batches, clean_label_batches = zip(*[batch for batch in train_loader])
    clean_images = torch.cat(clean_image_batches).numpy()
    clean_labels = torch.cat(clean_label_batches).numpy() 

    thermometer_images, _  = encoding(clean_images)
    if save:
        np.save('../thermometer_encoded_clean.npy', thermometer_images)

    encoded_set = torch.utils.data.TensorDataset(torch.from_numpy(thermometer_images), torch.from_numpy(clean_labels).long())
    clean_encoded_loader = torch.utils.data.DataLoader(encoded_set, batch_size=128, num_workers=16)

    adv_images_batches, adv_label_batches = zip(*[batch for batch in adv_loader])
    adv_images = torch.cat(adv_images_batches).numpy()
    adv_labels = torch.cat(adv_label_batches).numpy() 

    attacked_encoded, _  = encoding(adv_images)

    if save:
        np.save('../thermometer_encoded_adversarial.npy', attacked_encoded)

    adv_encoded_set = torch.utils.data.TensorDataset(torch.from_numpy(attacked_encoded), torch.from_numpy(adv_labels).long())
    adv_encoded_loader = torch.utils.data.DataLoader(adv_encoded_set, batch_size=128, num_workers=16)
    return clean_encoded_loader, adv_encoded_loader


