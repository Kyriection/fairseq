import copy 
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


__all__  = ['prune_model_l1', 'prune_model_random', 'prune_model_custom', 'remove_prune',
            'extract_mask', 'reverse_mask', 'check_sparsity', 'check_sparsity_dict', 'check_sparsity_overall']


# Pruning operation
def prune_model_l1(model, px):

    print('Apply Unstructured L1 Pruning Globally (all conv & linear layers)')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            parameters_to_prune.append((m,'weight'))
            if hasattr(m, 'bias'):
                if not m.bias == None:
                    parameters_to_prune.append((m,'bias'))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def prune_model_random(model, px):

    print('Apply Unstructured Random Pruning Globally (all conv & linear layers)')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            parameters_to_prune.append((m,'weight'))
            if hasattr(m, 'bias'):
                if not m.bias == None:
                    parameters_to_prune.append((m,'bias'))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )

def prune_model_custom(model, mask_dict):

    print('Pruning with custom mask (all conv & linear layers)')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            weight_mask_name = name+'.weight_mask'
            if weight_mask_name in mask_dict.keys():
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print('Can not find [{}] in mask_dict'.format(weight_mask_name))

            if hasattr(m, 'bias'):
                if not m.bias == None:
                    bias_mask_name = name+'.bias_mask'
                    if bias_mask_name in mask_dict.keys():
                        prune.CustomFromMask.apply(m, 'bias', mask=mask_dict[name+'.bias_mask'])
                    else:
                        print('Can not find [{}] in mask_dict'.format(bias_mask_name))

def remove_prune(model):
    
    print('Remove hooks for multiplying masks (all conv & linear layers)')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            prune.remove(m,'weight')
            if hasattr(m, 'bias'):
                if not m.bias == None:
                    prune.remove(m,'bias')


# Modify Mask
def extract_mask(model_dict):

    new_dict = {}
    for key in model_dict.keys():
        if '_mask' in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict

def reverse_mask(mask_dict):

    new_dict = {}
    for key in mask_dict.keys():

        new_dict[key] = 1 - mask_dict[key]

    return new_dict

# Calculate Sparsity
def check_sparsity(model):
    
    sum_element = 0
    zero_sum = 0

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            sum_element += float(m.weight.nelement())
            zero_sum += float(torch.sum(m.weight == 0))  

    remain_weight_ratie = 100*(1-zero_sum/sum_element)
    print('* remain weight ratio = {:.4f}%'.format(remain_weight_ratie))

    return remain_weight_ratie


# Calculate Sparsity
def check_sparsity_overall(model):
    
    sum_element = 0
    zero_sum = 0

    for name,m in model.named_modules():
        sum_element += float(m.weight.nelement())
        zero_sum += float(torch.sum(m.weight == 0))  

    remain_weight_ratie = 100*(1-zero_sum/sum_element)
    print('* remain weight ratio (overall) = {:.4f}%'.format(remain_weight_ratie))

    return remain_weight_ratie


def check_sparsity_dict(state_dict):
    
    sum_element = 0
    zero_sum = 0

    for key in state_dict.keys():
        if '_mask' in key:
            sum_element += float(state_dict[key].nelement())
            zero_sum += float(torch.sum(state_dict[key] == 0))  

    remain_weight_ratie = 100*(1-zero_sum/sum_element)
    print('* remain weight ratio = {:.4f}%'.format(remain_weight_ratie))

    return remain_weight_ratie
