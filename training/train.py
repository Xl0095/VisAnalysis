from transformers import get_linear_schedule_with_warmup
import torch
import random
import numpy as np
import pandas as pd
import time
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from torch.utils.data.dataset import Subset

from utils import display, my_model, my_dataset
from utils.log import write_log


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
"""
def flat_accuracy_f1(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    tp = tn = fp = fn = 0
    for i in range(pred_flat.shape[0]):
        if pred_flat[i] == labels_flat[i]:
            if pred_flat[i] == 0:
                tn += 1
            else:
                tp += 1
        else:
            if pred_flat[i] == 0:
                fn += 1
            else:
                fp += 1
    acc = (tp + tn) / pred_flat.shape[0]
    val_1cnt = sum(labels_flat)
    pred_1cnt = sum(pred_flat)
    return acc, tp, tn, fp, fn, pred_1cnt, val_1cnt
"""

# tensor dataset
def train(dataset, seed_val, log_file, res_file):
    # prepare dataloader

    # Number of training epochs. The BERT authors recommend between 2 and 4. 
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4
    cols = ['lr', 'k', 'epoch', 'Train. Loss', 'Valid. Loss', 'Valid. Accur.', 'Valid. macro F1', 'Valid. micro F1', 'Valid. weighted F1']
    lrs = [5e-5, 1e-4, 4e-4, 1e-3, 3e-3, 1e-2]
    all_stats = pd.DataFrame(columns=cols)
    for lr in lrs:
        print(f'===============learning rate: {lr}==============')
        kf = StratifiedKFold(n_splits=5, shuffle=True)

        for k, (train_index, val_index) in enumerate(kf.split(dataset, dataset[:][2])):
            print(f'--------FOLD {k}---------')
            train_dataset = Subset(dataset, train_index)
            val_dataset = Subset(dataset, val_index)
            train_dataloader, validation_dataloader = my_dataset.get_train_val_dataloader(train_dataset, val_dataset)
            model, optimizer = my_model.load_model_optimizer(lr)

            # select device
            if torch.cuda.is_available():
                device = torch.device('cuda')
                model.cuda()
                print('We will use the GPU:', torch.cuda.get_device_name(0))
            else:
                print('No GPU available, using the CPU instead.')
                device = torch.device('cpu')
            # device = torch.device('cpu')

            now_stats = training(model, optimizer, device, train_dataloader, validation_dataloader, seed_val, epochs, log_file)
            df_stats = pd.DataFrame(data=now_stats)
            df_stats.insert(0, 'k', [k] * epochs)
            df_stats.insert(0, 'lr', [lr] * epochs)
            all_stats = pd.concat([all_stats, df_stats], ignore_index=True)
    display.print_res(all_stats, res_file)



def training(model, optimizer, device, train_dataloader, validation_dataloader, seed_val, epochs, log_file):
    # Number of training epochs. The BERT authors recommend between 2 and 4. 
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    # epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # training
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        # print("")
        # print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        # print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0.0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                # print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        # print("")
        # print("  Average training loss: {0:.2f}".format(avg_train_loss))
        # print("  Training epcoh took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # print("")
        # print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_loss = 0
        nb_eval_steps = 0
        total_p_1 = 0
        total_v_1 = 0
        total_v_len = 0

        y_pred = []
        y_test = []

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            total_p_1 += sum(pred_flat)
            total_v_1 += sum(labels_flat)
            total_v_len += len(labels_flat)

            y_pred.extend(pred_flat)
            y_test.extend(labels_flat)
        

        # Report the final accuracy for this validation run.

        print(y_pred, y_test)
        acc = metrics.accuracy_score(y_test, y_pred)

        mi_f1 = metrics.f1_score(y_test, y_pred, average='micro')
        we_f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        ma_f1 = metrics.f1_score(y_test, y_pred, average='macro')

        print("    macro F1: {0:.4f}".format(ma_f1))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        # Measure how long the validation run took.
        # validation_time = format_time(time.time() - t0)
        
        # print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        # print("  Validation took: {:}".format(validation_time))
        write_log(log_file, "epoch {} -- pred 1 count, val 1 count, val length: {}, {}, {}\n".format(epoch_i + 1, total_p_1, total_v_1, total_v_len))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Train. Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': acc,
                'Valid. macro F1': ma_f1, 
                'Valid. micro F1': mi_f1,
                'Valid. weighted F1': we_f1
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    return training_stats
    # display.print_res(training_stats)