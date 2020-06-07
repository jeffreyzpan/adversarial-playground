import torch
import numpy as np
from art.attacks.evasion import CarliniLInfMethod
import art.defences as defences
from tqdm import tqdm

def gen_attacks(test_loader, classifier, attacks, epsilons, gpu_id_list='0,1,2,3', use_gpu=True): 

    adv_dict = {}
    test_labels = torch.tensor(test_loader.dataset.targets).long()

    # loop through list of attacks and generate adversarial images using the given method
    for attack_name, attack in zip(attacks.keys(), attacks.values()):

        print(attack_name)
        adv_list = [[] for i in range(len(epsilons))]
        for i, (inputs, target) in enumerate(tqdm(test_loader)):
            if use_gpu:
                inputs = inputs.cuda(f'cuda:{gpu_id_list}', non_blocking=True)
                target = target.cuda(f'cuda:{gpu_id_list}', non_blocking=True)
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

def cw_linf(classifier, test_loader, epsilons):

    adv_dict = {}
    adv_list = []

    test_image_batches, test_label_batches = zip(*[batch for batch in test_loader])
    test_images = torch.cat(test_image_batches).numpy()
    test_labels = torch.cat(test_label_batches).numpy()

    for epsilon in epsilons:
        print('running cw_linf with eps {}'.format(epsilon))
        attack = CarliniLInfMethod(classifier, eps=epsilon, confidence=0.5, max_iter=10000)
        adv_examples = attack.generate(test_images)
       
        adv_set = torch.utils.data.TensorDataset(torch.from_numpy(adv_examples), torch.from_numpy(test_labels))
        adv_loader = torch.utils.data.DataLoader(adv_set, batch_size=128, num_workers=16)
        adv_list.append(adv_loader)
        print('done')

    adv_dict['cw_Linf'] = adv_list
    
    return adv_dict

def gen_defences(adv_loader, attack_name, defences):
    
    def_adv_dict = {}

    # loop through list of defenses and generate defended images using the given method if method isn't adv. training based
    for defence_name, defence in zip(defences.keys(), defences.values()):
        
        print(defence_name)
        def_adv_list = []
        test_label_list = []
        for i, (adv_images, test_labels) in enumerate(tqdm(adv_loader)):
            adv_images = adv_images.numpy()
            test_labels = test_labels.numpy()
            
            adv_images = np.clip(adv_images, 0, 1)

            def_adv, _ = defence(adv_images)

            def_adv_list.append(def_adv)
            test_label_list.append(test_labels)

        #convert np array of defended images to PyTorch dataloader for CUDA validation later

        def_adv_set = torch.utils.data.TensorDataset(torch.from_numpy(np.concatenate(def_adv_list)), torch.from_numpy(np.concatenate(test_label_list)).long())
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


