import torch
import torch.nn.functional as F

def soft_f1_loss(predictions, num_classes, targets, epsilon=1e-7):
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    
    tp = torch.sum(predictions * targets_one_hot, dim=0)
    fp = torch.sum(predictions * (1 - targets_one_hot), dim=0)
    fn = torch.sum((1 - predictions) * targets_one_hot, dim=0)
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    
    return 1 - f1.mean()

def multi_attribute_f1_loss(predictions, targets, category_count, epsilon=1e-7):
    category_loss = 0
    for attr_idx, output in enumerate(predictions):

        # Create a mask to identify valid (non -1) values in the targets for the current attribute
        valid_mask = (targets[:, attr_idx] != -1)
        
        # If there are no valid entries for this attribute, skip it
        if valid_mask.sum() == 0:
            continue
        
        # Filter out invalid values in targets and corresponding predictions
        valid_targets = targets[valid_mask, attr_idx].long()
        valid_output = output[valid_mask]

        loss = soft_f1_loss(valid_output, valid_output.shape[1], valid_targets)
        
        category_loss += loss

    return category_loss / category_count
