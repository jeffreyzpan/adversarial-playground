import torch
import numpy as np
import art.defences as defences

def gen_attacks(test_images, test_labels, classifier, criterion, attacks): 

    adv_dict = {}

    # loop through list of attacks and generate adversarial images using the given method
    for attack_name, attack in zip(attacks.keys(), attacks.values()):

        print(attack_name)
        adv_test = attack.generate(x=test_images)

        #convert np array of adv. images to PyTorch dataloader for CUDA validation later
        adv_set = torch.utils.data.TensorDataset(torch.from_numpy(adv_test), torch.from_numpy(test_labels).long())
        adv_loader = torch.utils.data.DataLoader(adv_set, batch_size=128, num_workers=16)
        adv_dict[attack_name] = adv_loader
        print('done')

    return adv_dict

def gen_defences(test_images, adv_images, attack_name, test_labels, classifier, criterion, defences):
    
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

def thermometer_encoding(train_loader, adv_loader, thm_params):

    thermometer_encoding = defences.ThermometerEncoding(clip_values=(thm_params['clip_min'], thm_params['clip_max']), num_space=thm_params['num_space'], channel_index=thm_params['channel_index']) 
    
    print('Generating thermometer encoded images') 
    clean_image_batches, clean_label_batches = zip(*[batch for batch in train_loader])
    clean_images = torch.cat(clean_image_batches).numpy()
    clean_labels = torch.cat(clean_label_batches).numpy() 
    clean_images  = np.transpose(clean_images, (0, 2, 3, 1))

    thermometer_images, _  = thermometer_encoding(clean_images)
    thermometer_images = np.transpose(thermometer_images, (0, 3, 1, 2))

    encoded_set = torch.utils.data.TensorDataset(torch.from_numpy(thermometer_images), torch.from_numpy(clean_labels).long())
    clean_encoded_loader = torch.utils.data.DataLoader(encoded_set, batch_size=128, num_workers=16)

    adv_images_batches, adv_label_batches = zip(*[batch for batch in adv_loader])
    adv_images = torch.cat(adv_images_batches).numpy()
    adv_labels = torch.cat(adv_label_batches).numpy() 

    attacked_encoded, _  = thermometer_encoding(adv_images)
    attacked_encoded = np.transpose(attacked_encoded, (0, 3, 1, 2))

    adv_encoded_set = torch.utils.data.TensorDataset(torch.from_numpy(attacked_encoded), torch.from_numpy(adv_labels).long())
    adv_encoded_loader = torch.utils.data.DataLoader(adv_encoded_set, batch_size=128, num_workers=16)
    return clean_encoded_loader, adv_encoded_loader


