
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


#### Set Path ####
# go to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
root = '../'
root_code = '../code'
sys.path.append(root)
sys.path.append(root_code)
print('current work directory: ', os.getcwd())


def _to_list(x):
    """tensor / numpy / list -> plain python list of floats or ints (JSON friendly)."""
    if torch.is_tensor(x):
        x = x.tolist()
    return [v.item() if hasattr(v, 'item') else v for v in x]


def oneRun(run_dir, **params):
    """ Run one experiment (one seed). All outputs are written into `run_dir`. """
    run_dir = run_dir if run_dir.endswith('/') else run_dir + '/'
    os.makedirs(run_dir, exist_ok=True)

    ##### Default Setting #####
    ## Set input data path
    if 'data_dir' in params and params['data_dir'] is not None:
        data_path = params['data_dir']
    else:
        data_path = root + 'data/' + params.get('dataset', 'ag_news')
    print('\ndata_path: ', data_path)

    ## Set default hyperparameters
    n_labeled_per_class = 10        if 'n_labeled_per_class' not in params else params['n_labeled_per_class']
    bs = 8                          if 'bs' not in params else params['bs']
    ul_ratio = 10                   if 'ul_ratio' not in params else params['ul_ratio']
    lr = 2e-5                       if 'lr' not in params else params['lr']
    lr_linear = 1e-3                if 'lr_linear' not in params else params['lr_linear']

    # - semi-supervised
    weight_u_loss = 0.1             if 'weight_u_loss' not in params else params['weight_u_loss']
    load_mode = 'semi'              if 'load_mode' not in params else params['load_mode'] # semi, sup_baseline

    # - pseudo-labeling
    psl_threshold_h = 0.98          if 'psl_threshold_h' not in params else params['psl_threshold_h']
    labeling_mode = 'hard'          if 'labeling_mode' not in params else params['labeling_mode'] # hard, soft
    adaptive_threshold = False      if 'adaptive_threshold' not in params else params['adaptive_threshold'] # True, False

    # - ensemble
    num_nets = 2                    if 'num_nets' not in params else params['num_nets']
    cross_labeling = False          if 'cross_labeling' not in params else params['cross_labeling'] # True, False

    # - weight disagreement
    weight_disagreement = False     if 'weight_disagreement' not in params else params['weight_disagreement'] # 'True', False
    disagree_weight = 0.9           if 'disagree_weight' not in params else params['disagree_weight']   # fixed delta

    # - adaptive disagreement weight (NEW): delta_t is computed from an EMA of the nets' disagreement rate
    from utils.adaptive_delta import DisagreementEMA
    disagree_mode = 'fixed'         if 'disagree_mode' not in params else params['disagree_mode']   # 'fixed' (paper) | 'adaptive'
    delta_min = 0.6                 if 'delta_min' not in params else params['delta_min']
    delta_max = 0.9                 if 'delta_max' not in params else params['delta_max']
    delta_ema_momentum = 0.99       if 'delta_ema_momentum' not in params else params['delta_ema_momentum']
    delta_warmup_mass = 20.0        if 'delta_warmup_mass' not in params else params['delta_warmup_mass']
    variant = 'custom'              if 'variant' not in params else params['variant']
    keep_checkpoints = False        if 'keep_checkpoints' not in params else params['keep_checkpoints']
    dstate = DisagreementEMA(disagree_mode=disagree_mode, fixed_weight=disagree_weight, delta_min=delta_min,
                             delta_max=delta_max, momentum=delta_ema_momentum, warmup_mass=delta_warmup_mass)

    # - ema
    ema_mode = False                if 'ema_mode' not in params else params['ema_mode']
    ema_momentum = 0.9              if 'ema_momentum' not in params else params['ema_momentum']

    # - others
    seed = params['seed']
    device_idx = 0                  if 'device_idx' not in params else params['device_idx']
    val_interval = 25               if 'val_interval' not in params else params['val_interval']
    early_stop_tolerance = 10       if 'early_stop_tolerance' not in params else params['early_stop_tolerance']
    max_epoch = 10000               if 'max_epoch' not in params else params['max_epoch']
    max_step = 100000               if 'max_step' not in params else params['max_step']


    ## Set random seed and device
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # Check & set device
    if torch.cuda.is_available():
        if device_idx >= torch.cuda.device_count():
            device_idx = 0
        device = torch.device("cuda", device_idx)
        device_name = torch.cuda.get_device_name(device_idx)
        print('\nThere are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU-', device_idx, device_name)
    else:
        print('\nNo GPU available, using the CPU instead.')
        device = torch.device("cpu")
        device_name = 'cpu'


    #### Load Data ####
    from utils.dataloader import get_dataloader
    train_labeled_loader, train_unlabeled_loader, dev_loader, test_loader, n_classes = get_dataloader(data_path, n_labeled_per_class, bs, load_mode)
    print('n_classes: ', n_classes)

    class_names_path = os.path.join(data_path, 'classes.txt')
    if os.path.exists(class_names_path):
        class_names = [l.strip() for l in open(class_names_path) if l.strip()]
    else:
        class_names = [str(i) for i in range(n_classes)]


    ##### Model & Optimizer #####
    from models.netgroup import NetGroup

    net_arch = 'bert-base-uncased'
    netgroup = NetGroup(net_arch, num_nets, n_classes, device, lr, lr_linear)

    # Initialize EMA
    netgroup.train()
    if ema_mode:
        netgroup.init_ema(ema_momentum)


    ##### Training & Evaluation #####
    from utils.helper import format_time
    from criterions.criterions import ce_loss, consistency_loss
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support

    ## Evaluation
    @torch.no_grad() # no need to track gradients in validation
    def evaluation(loader):
        """Evaluate the ensemble (mean of softmax over nets) and every single net."""
        netgroup.eval()
        if ema_mode: # use ema model for evaluation
            netgroup.eval_ema()

        preds_ens, target_all = [], []
        preds_nets = [[] for _ in range(num_nets)]

        for batch in loader:
            b_labels = batch['label'].to(device)
            outs = netgroup.forward(batch['x'], b_labels)

            # take average probs from all nets
            probs = torch.mean(torch.softmax(torch.stack(outs), dim=2), dim=0)
            preds_ens.append(torch.argmax(probs, dim=1))
            target_all.append(b_labels)
            for i in range(num_nets):
                preds_nets[i].append(torch.argmax(outs[i], dim=1))

        preds_ens = torch.cat(preds_ens).cpu().numpy()
        target_all = torch.cat(target_all).cpu().numpy()
        preds_nets = [torch.cat(p).cpu().numpy() for p in preds_nets]

        cm = confusion_matrix(target_all, preds_ens, labels=list(range(n_classes)))
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_cw = np.where(cm.sum(1) > 0, np.diag(cm) / cm.sum(1), np.nan).round(3)

        res = {
            'acc': accuracy_score(target_all, preds_ens),
            'f1': f1_score(target_all, preds_ens, average='macro'),
            'acc_cw': acc_cw.tolist(),
            'acc_nets': [accuracy_score(target_all, p) for p in preds_nets],
            'f1_nets': [f1_score(target_all, p, average='macro') for p in preds_nets],
            # fraction of samples on which net 0 and net 1 predict different classes
            'net_disagree_rate': float(np.mean(preds_nets[0] != preds_nets[1])) if num_nets >= 2 else float('nan'),
            'confmat': cm,
            'preds_ens': preds_ens, 'preds_nets': preds_nets, 'target': target_all,
        }
        return res


    ## Training
    # Initialize variables
    t0 = time.time()
    step = 0
    best_acc = 0
    best_model_step = 0
    early_stop_count = 0
    pslt_global = 0
    psl_total_eval = 0
    psl_correct_eval = 0
    cw_psl_total, cw_psl_correct = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    cw_psl_total_accum, cw_psl_correct_accum = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    training_stats = []
    delta_trace = []
    early_stop_flag = False
    cw_avg_prob = (torch.ones(n_classes) / n_classes).to(device)    # estimate learning stauts of each class
    local_threshold = torch.zeros(n_classes, dtype=int)

    # delta in effect before the first step (paper value 0.9 until the EMA has enough evidence)
    delta_t = dstate.delta()

    # statistics on the agreement of the two nets' confident pseudo-labels, accumulated per eval interval
    def new_interval():
        return dict(n=0, agree=0, disagree=0, agree_correct=0, agree_same=0, dis_correct=0)
    istats = new_interval()

    ckpt_dir = run_dir + 'ckpt/'
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training
    netgroup.train()
    for epoch in range(max_epoch):
        if step > max_step:
            break
        if early_stop_flag:
            break
        for batch_label in train_labeled_loader:
            ## --- Evaluation: Check Performance on Validation set every val_interval batches ---##
            if step % val_interval == 0:
                res_val = evaluation(dev_loader)
                res_train = evaluation(train_labeled_loader)
                acc_val, f1_val = res_val['acc'], res_val['f1']
                acc_train, f1_train = res_train['acc'], res_train['f1']

                # restore training mode
                if ema_mode:
                    netgroup.train_ema()
                netgroup.train()

                print('>>Epoch %d Step %d acc_val %f f1_val %f acc_train %f f1_train %f psl_cor %d psl_totl %d pslt_global %f delta %.3f dis_ema %s' %
                        (epoch, step, acc_val, f1_val, acc_train, f1_train, psl_correct_eval, psl_total_eval, pslt_global, delta_t, ('%.3f' % dstate.rate()) if dstate.rate() is not None else 'n/a'),
                        'Tim {:}'.format(format_time(time.time() - t0)))

                # Record all statistics from this evaluation.
                acc_psl = (psl_correct_eval/psl_total_eval) if psl_total_eval > 0 else None
                nan = float('nan')
                n_u = max(istats['n'], 1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    cw_psl_acc_eval = (cw_psl_correct_eval / cw_psl_total_eval).tolist()
                    cw_psl_acc_accum = (cw_psl_correct_accum / cw_psl_total_accum).tolist()

                training_stats.append(
                    {   'step': step,
                        'delta_t': delta_t,
                        'disagree_ema': dstate.rate() if dstate.rate() is not None else nan,
                        'acc_val': acc_val,
                        'f1_val': f1_val,
                        'acc_train': acc_train,
                        'f1_train': f1_train,
                        'psl_correct': psl_correct_eval,
                        'psl_total': psl_total_eval,
                        'acc_psl': acc_psl,
                        'pslt_global': pslt_global,
                        # --- new: ensemble vs single nets, and divergence of the nets (dev set) ---
                        'acc_val_net0': res_val['acc_nets'][0],
                        'acc_val_net1': res_val['acc_nets'][1] if num_nets >= 2 else nan,
                        'dev_net_disagree_rate': res_val['net_disagree_rate'],
                        # --- new: agreement / disagreement of confident pseudo-labels (this eval interval) ---
                        'frac_agree': istats['agree'] / n_u,
                        'frac_disagree': istats['disagree'] / n_u,
                        'frac_neither': 1 - (istats['agree'] + istats['disagree']) / n_u,
                        'psl_acc_agree': istats['agree_correct'] / istats['agree'] if istats['agree'] > 0 else nan,
                        'psl_acc_disagree': istats['dis_correct'] / istats['disagree'] if istats['disagree'] > 0 else nan,
                        'agree_same_class_rate': istats['agree_same'] / istats['agree'] if istats['agree'] > 0 else nan,
                        # --- classwise lists (stored as JSON strings in the csv) ---
                        'cw_acc_train': json.dumps(res_train['acc_cw']),
                        'cw_acc_val': json.dumps(res_val['acc_cw']),
                        'cw_avg_prob': json.dumps(_to_list(cw_avg_prob)),
                        'local_threshold': json.dumps(_to_list(local_threshold)),
                        'cw_psl_total_eval': json.dumps(_to_list(cw_psl_total_eval)),
                        'cw_psl_correct_eval': json.dumps(_to_list(cw_psl_correct_eval)),
                        'cw_psl_acc_eval': json.dumps(cw_psl_acc_eval),
                        'cw_psl_total_accum': json.dumps(_to_list(cw_psl_total_accum)),
                        'cw_psl_correct_accum': json.dumps(_to_list(cw_psl_correct_accum)),
                        'cw_psl_acc_accum': json.dumps(cw_psl_acc_accum),
                    })

                # check classwise psl accuracy and total psl accuracy for the current eval
                print('cw_psl_eval: ', cw_psl_total_eval.tolist(), cw_psl_correct_eval.tolist())
                print('psl_acc: ', round((psl_correct_eval/psl_total_eval),3), end=' ') if psl_total_eval > 0 else print('psl_acc: None', end=' ')
                print('cw_psl_acc: ', cw_psl_acc_eval)

                # Early stopping & Save best model
                # - best criterion: acc_val
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_model_step = step
                    early_stop_count = 0
                    netgroup.save_model(ckpt_dir, 'model', ema_mode=ema_mode)
                else:
                    early_stop_count += 1
                    if early_stop_count >= early_stop_tolerance:
                        early_stop_flag = True
                        print('Early stopping trigger at step: ', step)
                        print('Best model at step: ', best_model_step)
                        break

                # initialize pseudo labels evaluation
                psl_total_eval, psl_correct_eval = 0, 0
                cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
                istats = new_interval()


            ## --- Training --- ##
            step += 1

            step_deltas = []   # delta values used in the unlabeled batches of this step

            ## Process Labeled Data
            x_lb, y_lb = batch_label['x_w'], batch_label['label']
            # forward pass
            outs_x_lb = netgroup.forward(x_lb, y_lb.to(device))
            # compute loss for labeled data
            sup_loss_nets = [ce_loss(outs_x_lb[i], y_lb.to(device)) for i in range(num_nets)]
            # update netgorup from loss of labeled data
            netgroup.update(sup_loss_nets)
            if ema_mode:
                netgroup.update_ema()


            ## Process Unlabeled Data
            if load_mode == 'semi':
                for _ in range(ul_ratio):
                    # unlabeled data input for each round/batch
                    try:
                        batch_unlabel = next(data_iter_unl)
                    except:
                        data_iter_unl = iter(train_unlabeled_loader)
                        batch_unlabel = next(data_iter_unl)

                    x_ulb_w, x_ulb_s = batch_unlabel['x_w'], batch_unlabel['x_s']

                    # forward pass
                    outs_x_ulb_s_nets = netgroup.forward(x_ulb_s)
                    with torch.no_grad(): # stop gradient for weak augmentation brach
                        outs_x_ulb_w_nets = netgroup.forward(x_ulb_w)

                    ## Generate pseudo labels and masks for all nets in one batch of unlabeled data
                    pseudo_labels_nets = []
                    u_psl_masks_nets = []

                    for i in range(num_nets):
                        ## generate pseudo labels
                        logits_x_ulb_w = outs_x_ulb_w_nets[i]
                        if labeling_mode=='soft':
                            pseudo_labels_nets.append(torch.softmax(logits_x_ulb_w, dim=-1))
                        else:
                            max_idx = torch.argmax(logits_x_ulb_w, dim=-1)
                            pseudo_labels_nets.append(F.one_hot(max_idx, num_classes=n_classes).to(device))

                        ## compute mask for pseudo labels
                        probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
                        max_probs, max_idx = torch.max(probs_x_ulb_w, dim=-1)
                        if not adaptive_threshold:
                            # fixed hard threshold
                            pslt_global = psl_threshold_h
                            u_psl_masks_nets.append(max_probs >= pslt_global)
                        else:
                            # adaptive local threshold
                            pslt_global = psl_threshold_h
                            cw_avg_prob = cw_avg_prob * ema_momentum + torch.mean(probs_x_ulb_w, dim=0) * (1-ema_momentum)
                            local_threshold = cw_avg_prob / torch.max(cw_avg_prob,dim=-1)[0]
                            u_psl_mask = max_probs.ge(pslt_global * local_threshold[max_idx])
                            u_psl_masks_nets.append(u_psl_mask)

                    ## (info only) agreement statistics of the two nets' confident pseudo-labels
                    if num_nets >= 2:
                        with torch.no_grad():
                            gt_all = batch_unlabel['label'].to(device)
                            m0, m1 = u_psl_masks_nets[0], u_psl_masks_nets[1]
                            pl0 = torch.argmax(outs_x_ulb_w_nets[0], dim=-1)
                            pl1 = torch.argmax(outs_x_ulb_w_nets[1], dim=-1)
                            agree_m = torch.logical_and(m0, m1)
                            dis_m = torch.logical_xor(m0, m1)
                            conf_pl = torch.where(m0, pl0, pl1)   # label of the confident net (disagree set)
                            istats['n'] += gt_all.numel()
                            istats['agree'] += agree_m.sum().item()
                            istats['disagree'] += dis_m.sum().item()
                            istats['agree_correct'] += ((pl0 == gt_all) & agree_m).sum().item()
                            istats['agree_same'] += ((pl0 == pl1) & agree_m).sum().item()
                            istats['dis_correct'] += ((conf_pl == gt_all) & dis_m).sum().item()
                            # EMA-adaptive disagreement weight: update with this batch, then use the new delta
                            dstate.update(agree_m.sum().item(), dis_m.sum().item())
                    delta_t = dstate.delta()
                    step_deltas.append(delta_t)

                    ## Compute loss for unlabeled data for all nets
                    total_unsup_loss_nets = []
                    for i in range(num_nets):
                        if cross_labeling:
                            # cross labeling and masking
                            pseudo_label = pseudo_labels_nets[(i+1)%num_nets]
                            u_psl_mask = u_psl_masks_nets[(i+1)%num_nets]
                        else:
                            # vanilla labeling
                            pseudo_label = pseudo_labels_nets[i]
                            u_psl_mask = u_psl_masks_nets[i]

                        if weight_disagreement == 'True':
                            # masks for disagreement and agreement across nets, derived from the confident pseudo-label masks.
                            # delta_t is the (possibly lr-adaptive) disagreement weight for this step.
                            disagree_mask = torch.logical_xor(u_psl_masks_nets[(i)%num_nets], u_psl_masks_nets[(i+1)%num_nets])
                            agree_mask = torch.logical_and(u_psl_masks_nets[(i)%num_nets], u_psl_masks_nets[(i+1)%num_nets])
                            disagree_weight_masked = delta_t * disagree_mask + (1-delta_t) * agree_mask
                        elif weight_disagreement == 'ablation_baseline':
                            disagree_mask = torch.logical_xor(u_psl_masks_nets[(i)%num_nets], u_psl_masks_nets[(i+1)%num_nets])
                            agree_mask = torch.logical_and(u_psl_masks_nets[(i)%num_nets], u_psl_masks_nets[(i+1)%num_nets])
                            disagree_weight_masked = 0.5 * disagree_mask + (1-0.5) * agree_mask
                        else:
                            disagree_weight_masked = None

                        # compute loss for unlabeled data
                        unsup_loss = consistency_loss(outs_x_ulb_s_nets[i], pseudo_label, loss_type='ce', mask=u_psl_mask, disagree_weight_masked=disagree_weight_masked)    # loss_type: 'ce' or 'mse'

                        # compute total loss for unlabeled data
                        total_unsup_loss = weight_u_loss * unsup_loss
                        total_unsup_loss_nets.append(total_unsup_loss)

                    # update netgorup from loss of unlabeled data
                    netgroup.update(total_unsup_loss_nets)
                    if ema_mode:
                        netgroup.update_ema()


                    #### -Info: for now we the last net in the group for info
                    ## Check the total and correct number of pseudo-labels
                    batch_unlabel['label'] = batch_unlabel['label'].to(device)
                    gt_labels_u = batch_unlabel['label'][u_psl_mask].to(device)
                    psl_total = torch.sum(u_psl_mask).item()

                    u_label_psl = pseudo_label[u_psl_mask]
                    u_label_psl_hard = torch.argmax(u_label_psl, dim=-1)
                    psl_correct = torch.sum(u_label_psl_hard == gt_labels_u).item()

                    psl_total_eval +=  psl_total
                    psl_correct_eval += psl_correct

                    # Check class-wise total and correct number of pseudo-labels
                    cw_psl_total = torch.bincount(u_label_psl_hard, minlength=n_classes).to('cpu')
                    cw_psl_correct = torch.bincount(u_label_psl_hard[u_label_psl_hard == gt_labels_u], minlength=n_classes).to('cpu')

                    cw_psl_total_eval += cw_psl_total
                    cw_psl_correct_eval += cw_psl_correct

                    cw_psl_total_accum += cw_psl_total
                    cw_psl_correct_accum += cw_psl_correct

            # record delta (mean over this step's unlabeled batches) and the disagreement EMA
            r_now = dstate.rate()
            delta_trace.append({'step': step,
                                'delta_t': float(np.mean(step_deltas)) if step_deltas else delta_t,
                                'disagree_ema': r_now if r_now is not None else float('nan')})


    train_time = time.time() - t0
    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(train_time)))


    ##### Results: load the best model, evaluate on the test set, and save everything #####
    netgroup.load_model(ckpt_dir, 'model', ema_mode=ema_mode)
    res_test = evaluation(test_loader)

    pd.set_option('display.precision', 4)

    # 1) training statistics (one row per evaluation) and the per-step lr / delta trace
    df_stats = pd.DataFrame(training_stats).set_index('step')
    df_stats.to_csv(run_dir + 'training_statistics.csv')
    pd.DataFrame(delta_trace).to_csv(run_dir + 'delta_trace.csv', index=False)
    print('Save training statistics in: ', run_dir + 'training_statistics.csv')

    # 2) confusion matrix (raw counts; rows = true class, columns = predicted class)
    cm = res_test['confmat']
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(run_dir + 'confusion_matrix.csv')

    # 3) per-class precision / recall / F1 on the test set
    p_, r_, f_, s_ = precision_recall_fscore_support(res_test['target'], res_test['preds_ens'],
                                                     labels=list(range(n_classes)), zero_division=0)
    pd.DataFrame({'class': class_names, 'precision': p_, 'recall': r_, 'f1': f_, 'support': s_}
                 ).to_csv(run_dir + 'per_class_metrics.csv', index=False)

    # 4) test predictions (for significance tests such as McNemar)
    pred_df = pd.DataFrame({'true': res_test['target'], 'pred_ensemble': res_test['preds_ens']})
    for i in range(num_nets):
        pred_df['pred_net%d' % i] = res_test['preds_nets'][i]
    pred_df.to_csv(run_dir + 'test_predictions.csv', index_label='idx')

    # 5) final metrics + config
    final_metrics = {
        'variant': variant, 'seed': seed,
        'test_acc': res_test['acc'], 'test_f1': res_test['f1'],
        'test_acc_nets': res_test['acc_nets'], 'test_f1_nets': res_test['f1_nets'],
        'test_net_disagree_rate': res_test['net_disagree_rate'],
        'best_step': best_model_step, 'best_dev_acc': best_acc,
        'last_step': step, 'stopped_early': bool(early_stop_flag),
        'delta_at_best_step': next((r['delta_t'] for r in delta_trace if r['step'] == max(best_model_step, 1)), delta_t),
        'delta_mean_until_best': float(np.mean([r['delta_t'] for r in delta_trace if r['step'] <= max(best_model_step, 1)])) if delta_trace else delta_t,
        'delta_mean_all': float(np.mean([r['delta_t'] for r in delta_trace])) if delta_trace else delta_t,
        'delta_at_last_step': delta_t,
        'train_time_sec': round(train_time, 1), 'device': device_name,
        'test_acc_cw': res_test['acc_cw'], 'class_names': class_names,
    }
    json.dump(final_metrics, open(run_dir + 'final_metrics.json', 'w'), indent=2, default=str)
    config = dict(params)
    config.update({'resolved': dict(disagree_mode=disagree_mode, disagree_weight=disagree_weight,
                                    delta_min=delta_min, delta_max=delta_max,
                                    delta_ema_momentum=delta_ema_momentum, delta_warmup_mass=delta_warmup_mass,
                                    lr=lr, bs=bs, ul_ratio=ul_ratio,
                                    n_labeled_per_class=n_labeled_per_class, psl_threshold_h=psl_threshold_h,
                                    weight_u_loss=weight_u_loss, ema_mode=ema_mode, ema_momentum=ema_momentum,
                                    val_interval=val_interval, early_stop_tolerance=early_stop_tolerance),
                   'torch': torch.__version__, 'finished': time.strftime("%Y%m%d-%H%M%S")})
    json.dump(config, open(run_dir + 'config.json', 'w'), indent=2, default=str)

    print('\nBest_step: ', best_model_step, '\nbest_test_acc: ', res_test['acc'], '\nbest_test_f1: ', res_test['f1'])

    # remove checkpoints (two BERT-base nets are large) unless asked to keep them
    if not keep_checkpoints:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    return final_metrics


##### multiRun ######
def multiRun(experiment_home=None, num_runs=3, seeds_list=None, resume=True, **params):
    """Run several seeds sequentially; each seed writes to <experiment_home>/seed_<seed>/ ."""
    import statistics

    if experiment_home is None:
        experiment_home = root + 'experiment/temp'
    os.makedirs(experiment_home, exist_ok=True)
    seeds_list = list(range(num_runs)) if seeds_list is None else list(seeds_list)

    results = []
    for seed in seeds_list:
        run_dir = os.path.join(experiment_home, 'seed_%d' % seed) + '/'
        done_file = run_dir + 'final_metrics.json'
        if resume and os.path.exists(done_file):
            print('seed %d already finished, loading %s' % (seed, done_file))
            results.append(json.load(open(done_file)))
            continue
        results.append(oneRun(run_dir, seed=seed, **params))

    accs = [r['test_acc'] for r in results]
    f1s = [r['test_f1'] for r in results]
    final = {'variant': params.get('variant', 'custom'), 'n_seeds': len(results),
             'seeds': seeds_list,
             'acc_mean': 100*statistics.mean(accs), 'acc_std': 100*statistics.pstdev(accs),
             'f1_mean': 100*statistics.mean(f1s), 'f1_std': 100*statistics.pstdev(f1s)}
    if len(results) > 1:   # (single-seed array jobs: use make_figures.py for the aggregate)
        pd.DataFrame([final]).to_csv(os.path.join(experiment_home, 'summary_avgrun.csv'), index=False)
    print('\nMean±std acc: %.2f ± %.2f | macro-F1: %.2f ± %.2f' %
          (final['acc_mean'], final['acc_std'], final['f1_mean'], final['f1_std']))
    return final
