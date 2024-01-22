import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf
from tqdm import tqdm
import torch.distributed as dist

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,test_dataloader):

        self.args = args

        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.model = model

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.epochs_val = self.args.epochs_val
        self.start_val = self.args.start_val
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = args.save_dir
        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self,rank):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            self.train_dataloader.sampler.set_epoch(epoch)
            result = self._train_epoch(epoch)

            best = False

            if epoch % self.epochs_val== 0 and epoch>self.start_val : #validation 
                val_result = self._val_epoch(rank, result)
                test_result = self._test_epoch(rank, result)
                # save logged informations into log dict
                log = {'epoch': epoch}
                log.update(val_result)
                log.update(test_result)
                self._record_best(log)

                # print logged informations to the screen
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                
                if self.mnt_mode != 'off':
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                            self.early_stop))
                        break

                    
            if epoch % self.save_period == 0 and rank==0:
                self._save_checkpoint(epoch, save_best=best)
        if rank == 0:
            self._print_best()
            self._print_best_to_file()
        
    def test(self,rank):
        log = {}
        
        if not os.path.exists(os.path.join(self.checkpoint_dir, 'reports')) and rank==0:
            os.mkdir(os.path.join(self.checkpoint_dir, 'reports'))
        
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.test_dataloader)):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                for i in range(len(reports)):
                    with open(os.path.join(self.checkpoint_dir, 'reports', images_id[i]) + '.txt','w') as f:
                        content = {}
                        content['predict'] = reports[i]
                        content['target'] = ground_truths[i]
                        f.write(str(content))
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        print('Results in test set')
        for key, value in log.items():
            print('\t{:15s}: {}'.format(str(key), value))

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):

        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader)


    def _train_epoch(self, rank):
        dist.barrier()
        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.train_dataloader, desc='Training', mininterval=300)):
            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
            output = self.model(images, reports_ids,  mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {'train_loss': train_loss / len(self.train_dataloader)}
        self.lr_scheduler.step()

        return log
    
    def _val_epoch(self, rank, log):
        dist.barrier()
        self.model.eval()
        with torch.no_grad():
            val_gts_ids, val_res_ids = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.val_dataloader,desc='Validating', mininterval=300)):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                # print('start eval...')
                output = self.model(images,  mode='sample')
                
                val_res_ids.append(output)    #predict
                val_gts_ids.append(reports_ids) #ground truth
            
            val_res_ids = distributed_concat(torch.cat(val_res_ids,dim=0),len(self.val_dataloader.dataset)).cpu().numpy()
            val_gts_ids = distributed_concat(torch.cat(val_gts_ids,dim=0),len(self.val_dataloader.dataset)).cpu().numpy()

            val_gts, val_res = self.model.module.tokenizer.decode_batch(val_gts_ids[:,1:]), self.model.module.tokenizer.decode_batch(val_res_ids)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
        return log
            
    def _test_epoch(self, rank, log):
        dist.barrier()
        self.model.eval()
        with torch.no_grad():
            test_gts_ids, test_res_ids = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.test_dataloader,desc='Testing', mininterval=300)):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images,  mode='sample')

                test_res_ids.append(output)    #predict
                test_gts_ids.append(reports_ids) #ground truth
            test_res_ids = distributed_concat(torch.cat(test_res_ids,dim=0),len(self.test_dataloader.dataset)).cpu().numpy()
            test_gts_ids = distributed_concat(torch.cat(test_gts_ids,dim=0),len(self.test_dataloader.dataset)).cpu().numpy()

            test_gts, test_res = self.model.module.tokenizer.decode_batch(test_gts_ids[:,1:]), self.model.module.tokenizer.decode_batch(test_res_ids)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})


        return log

    
def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]
