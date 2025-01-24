import numpy as np
import torch

from tqdm import tqdm
from utils import AverageMeter
from metrics import MetronAtK

def save_history(train_loss_list, val_score_list, save_path):
    history = {}

    history['train_loss'] = train_loss_list
    history['val_score'] = val_score_list

    np.save(save_path, history)

def train(args, model, sample_generator, optimizer):
    start_epoch = 0
    global_step = 0
    
    train_losses_avg = []
    val_score_avg = []
    
    val_data = sample_generator.evaluate_validation_data
    metron = MetronAtK(top_k=10)
    criterion = torch.nn.BCELoss()
    minimum_val_score = 0
    
    ## training stage
    for epoch in range(start_epoch, args.epochs):
        # model train mode
        model.train()
        train_dataloader = sample_generator.instance_a_train_loader() # for negative sampling
        train_losses = AverageMeter()
        train_t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    
        for batch_id, batch in train_t:
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
    
            # to cuda
            user = user.to('cuda')
            item = item.to('cuda')
            rating = rating.to('cuda')
    
            # calculate loss
            ratings_pred = model(user, item)
            loss = criterion(ratings_pred.view(-1), rating)
            train_losses.update(loss.item())
            
            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # print tqdm
            print_loss = round(loss.item(), 4)
            train_t.set_postfix_str("Train loss : {}".format(print_loss))
    
        # record train losses
        train_losses_avg.append(train_losses.avg)
    
        # evaluation stage
        print("evaluation steps.....")
        
        model.eval()
        with torch.no_grad():
            test_users, test_items = val_data[0], val_data[1]
            negative_users, negative_items = val_data[2], val_data[3]
        
            # to cuda
            test_users = test_users.to('cuda')
            test_items = test_items.to('cuda')
            negative_users = negative_users.to('cuda')
            negative_items = negative_items.to('cuda')
        
            # batch inference
            test_scores = []
            negative_scores = []
            bs = args.batch_size
        
            for start_idx in range(0, len(test_users), bs):
                end_idx = min(start_idx + bs, len(test_users))
                batch_test_users = test_users[start_idx:end_idx]
                batch_test_items = test_items[start_idx:end_idx]
                test_scores.append(model(batch_test_users, batch_test_items))
            for start_idx in tqdm(range(0, len(negative_users), bs)):
                end_idx = min(start_idx + bs, len(negative_users))
                batch_negative_users = negative_users[start_idx:end_idx]
                batch_negative_items = negative_items[start_idx:end_idx]
                negative_scores.append(model(batch_negative_users, batch_negative_items))
            test_scores = torch.concatenate(test_scores, dim=0)
            negative_scores = torch.concatenate(negative_scores, dim=0)
        
            # to cpu
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        
            metron.subjects = [test_users.data.view(-1).tolist(),
                               test_items.data.view(-1).tolist(),
                               test_scores.data.view(-1).tolist(),
                               negative_users.data.view(-1).tolist(),
                               negative_items.data.view(-1).tolist(),
                               negative_scores.data.view(-1).tolist()]
        
            # calculate score
            hit_ratio, ndcg = metron.cal_hit_ratio(), metron.cal_ndcg()
        
            # calculate average score
            avg_score = (hit_ratio + ndcg) / 2
            val_score_avg.append(avg_score)
    
            # save best model
            if avg_score > minimum_val_score:
                print('improve validation score!! so model save {} -> {}'.format(minimum_val_score, avg_score))
                minimum_val_score = avg_score
                torch.save(model.state_dict(), args.model_save_path)
                
            # save history
            save_history(train_losses_avg, val_score_avg, save_path=args.model_save_path.replace('.pth', '.npy'))