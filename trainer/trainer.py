import torch
from torch.nn import MSELoss
import time
import math
import logging

from util.metrics_util import calc_qerror


def get_batch_data_to_device(batch, device):
    batch_feat = batch[0]
    batch_label = batch[1]
    
    feats_to_device = {}
    for key, tensor in batch_feat.items():
        feats_to_device[key] = tensor.to(device)
        
    labels_to_device = batch_label.to(device)
    return feats_to_device, labels_to_device


def evaluate(model, val_dataloader, device, args, criterion=None):
    model.eval()
    val_pred_list = []
    val_label_list = []
    tol_loss = 0
    if not criterion:
        criterion = MSELoss()
    with torch.no_grad():
        for batch in val_dataloader:
            features, label = get_batch_data_to_device(batch, device)
            
            val_preds = model(features)
            val_preds = val_preds.squeeze()
            loss = criterion(val_preds, label)
            tol_loss += loss.item()

            val_pred_list.extend(val_preds.cpu().detach().numpy().flatten().tolist())
            val_label_list.extend(label.cpu().detach().numpy().flatten().tolist())

    avg_loss = tol_loss / len(val_dataloader)

    if args.log_label:
        val_pred_list = [min(math.pow(math.e, i) - 1, 1.0) for i in val_pred_list]
        val_label_list = [math.pow(math.e, i) - 1 for i in val_label_list]
    scores = calc_qerror(val_pred_list, val_label_list)
    
    logging.info(f'Eval:  Avg Loss: {avg_loss}, Q-Error: {scores}')
    return avg_loss, val_pred_list, val_label_list, scores


def train(model, train_dataloader, val_dataloader, \
          args, criterion=None, optimizer=None, scheduler=None, model_save_path=None):

    device, epochs, clip_size, lr = args.device, args.epochs, args.clip_size, args.lr
    
    model.to(device)

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.7)
    if not criterion:
        criterion = MSELoss()
    
    train_loss_list = []
    val_loss_list = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_pred_list = []
        train_label_list = []

        model.train()
        logging.info(f"---- Epoch {epoch + 1} / {epochs} ---- ")

        i = 0
        for batch in train_dataloader:
            if i == 0:
                i += 1
                continue
            
            features, label = get_batch_data_to_device(batch, device)
            optimizer.zero_grad()

            train_preds = model(features)
            train_preds = train_preds.squeeze()

            loss = criterion(train_preds, label)
            loss.backward()
            if clip_size is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)
            optimizer.step()
            epoch_loss += loss.item()

            train_pred_list.extend(train_preds.detach().cpu().numpy().flatten().tolist())
            train_label_list.extend(label.cpu().detach().numpy().flatten().tolist())
            
        train_loss_list.append(epoch_loss / len(train_dataloader))
        
        if args.log_label:
            train_pred_list = [min(math.pow(math.e, i) - 1, 1.0) for i in train_pred_list]
            train_label_list = [math.pow(math.e, i) - 1 for i in train_label_list]
        train_scores = calc_qerror(train_pred_list, train_label_list)
        
        logging.info(f"Train: Avg Loss: {train_loss_list[-1]}, Q-Error: {train_scores}")
        
        # Validation step
        val_loss, val_pred_list, val_label_list, val_scores = evaluate(model, val_dataloader, device, args, criterion)
        val_loss_list.append(val_loss)
        
        logging.info(f"Epoch {epoch + 1} finished. Time: {time.time() - epoch_start_time}s\n")
        scheduler.step()

    if model_save_path:
        torch.save({
            'model': model.state_dict(),
            'args' : args
        }, model_save_path)
        logging.info(f"Model saved at {model_save_path}")
        
    return train_loss_list, train_pred_list, train_label_list, train_scores, \
           val_loss_list, val_pred_list, val_label_list, val_scores
           
           
def group_evaluate(model, val_dataloader, args, val_items):
    device = args.device
    model.eval()
    val_pred_list = []
    val_label_list = []
    with torch.no_grad():
        for batch in val_dataloader:
            features, label = get_batch_data_to_device(batch, device)
            
            val_preds = model(features)
            val_preds = val_preds.squeeze()

            val_pred_list.extend(val_preds.cpu().detach().numpy().flatten().tolist())
            val_label_list.extend(label.cpu().detach().numpy().flatten().tolist())

    if args.log_label:
        val_pred_list = [min(math.pow(math.e, i) - 1, 1.0) for i in val_pred_list]
        val_label_list = [math.pow(math.e, i) - 1 for i in val_label_list]
    scores = calc_qerror(val_pred_list, val_label_list)
    
    
    query_type2lables = {}
    query_type2preds = {}
    for i, it in enumerate(val_items):
        # assert abs(it["label"] - val_label_list[i]) < 0.0001
        # query_type = it["query_type"]
        query_type = it["group_num"] if "group_num" in it else it["query_type"]
        if query_type not in query_type2lables:
            query_type2lables[query_type] = []
            query_type2preds[query_type] = []
        query_type2lables[query_type].append(val_label_list[i])
        query_type2preds[query_type].append(val_pred_list[i])
    query_type2scores = {}
    for query_type in query_type2lables:
        score = calc_qerror(query_type2lables[query_type], query_type2preds[query_type])
        query_type2scores[query_type] = score
        logging.info(f'{query_type}: {score}')
    query_type2scores["all"] = scores
    return query_type2scores

def template_evaluate(model, val_dataloader, args, val_items):
    device = args.device
    model.eval()
    val_pred_list = []
    val_label_list = []
    with torch.no_grad():
        for batch in val_dataloader:
            features, label = get_batch_data_to_device(batch, device)
            
            val_preds = model(features)
            val_preds = val_preds.squeeze()

            val_pred_list.extend(val_preds.cpu().detach().numpy().flatten().tolist())
            val_label_list.extend(label.cpu().detach().numpy().flatten().tolist())

    if args.log_label:
        val_pred_list = [min(math.pow(math.e, i) - 1, 1.0) for i in val_pred_list]
        val_label_list = [math.pow(math.e, i) - 1 for i in val_label_list]
    scores = calc_qerror(val_pred_list, val_label_list)
    
    
    query_type2lables = {}
    query_type2preds = {}
    for i, it in enumerate(val_items):
        # assert abs(it["label"] - val_label_list[i]) < 0.0001
        # query_type = it["query_type"]
        query_type = it["query_type"] + "#" + str(it["query"].nr)
        if query_type not in query_type2lables:
            query_type2lables[query_type] = []
            query_type2preds[query_type] = []
        query_type2lables[query_type].append(val_label_list[i])
        query_type2preds[query_type].append(val_pred_list[i])
    query_type2scores = {}
    for query_type in query_type2lables:
        score = calc_qerror(query_type2lables[query_type], query_type2preds[query_type])
        query_type2scores[query_type] = score
        logging.info(f'{query_type}: {score}')
    query_type2scores["all"] = scores
    return query_type2scores