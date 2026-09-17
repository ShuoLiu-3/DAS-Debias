import copy
import torch
import torch.nn.functional as F
import logging
logger = logging.getLogger(name='logger')


def random_l2_noise(adv_embeddings, epsilon):

    _, batch_size, dim = adv_embeddings.shape
    noise = torch.randn(1, batch_size, dim, device=adv_embeddings.device)
    noise_norm = torch.norm(noise, p=2, dim=2, keepdim=True)
    normalized_noise = noise / noise_norm
    scaling_factors = torch.rand(1, batch_size, 1, device=adv_embeddings.device) * epsilon
    random_noise = normalized_noise * scaling_factors
    return random_noise



def pgd_attack_general(model, embeddings, epsilon=1, eps_iter=0.1, num_iter=10, device=None,
               loss_type='KL'):

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    adv_embeddings = embeddings.detach().clone().to(dtype=torch.float32, device=device)
    outputs = model(adv_embeddings)
    ori_probabilities = torch.softmax(outputs, dim=-1)
    num_classes = ori_probabilities.size(-1)
    target_dist = torch.ones_like(ori_probabilities) / num_classes

    delta = random_l2_noise(adv_embeddings, epsilon)
    delta.requires_grad = True

    best_delta = delta.detach().clone()
    best_loss = -1e10
    with torch.enable_grad():
        for cur_iter in range(num_iter):
            outputs = model(adv_embeddings + delta)
            probabilities = torch.softmax(outputs, dim=-1)
            log_probabilities = torch.log_softmax(outputs, dim=-1)
            if loss_type == 'KL':
                loss = -F.kl_div(log_probabilities, target_dist, reduction='batchmean')
            if loss.item() > best_loss:
                best_loss = loss.item()
                best_delta = delta.detach().clone()
            if best_loss > -0.03:
                break

            model.zero_grad()
            loss.backward(retain_graph=True)

            delta = delta + eps_iter * delta.grad.data.sign()
            delta_norm = torch.norm(delta, p=2, dim=1, keepdim=True)
            factor = torch.min(torch.ones_like(delta_norm), epsilon / delta_norm)
            delta = delta * factor
            delta.detach_()
            delta.requires_grad = True

    new_embeddings = adv_embeddings + best_delta
    outputs = model(new_embeddings)
    probabilities = torch.softmax(outputs, dim=-1)
    loss = -F.kl_div(probabilities.log(), target_dist, reduction='batchmean')
    return new_embeddings.detach(), loss, cur_iter + 1