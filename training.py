from os.path import join

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

class Trainer(object):
    def __init__(self, optimizer, model, train_loader,
                 val_loader, save_dir, device, clip,
                 print_freq, ckpt_freq, patience, copy):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        self.device = device
        self.clip = clip
        self.print_freq = print_freq
        self.ckpt_freq = ckpt_freq
        self.patience = patience
        self.model_is_copy = copy

        self.step = 0
        self.epoch = 1

        self.current_p = 0
        self.best_val = 1e18
        self.current_val_loss = 1e18

    def validate(self):
        # cal val loss
        self.model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for srcs, targets in self.val_loader:
                srcs, src_lens, tgt = srcs
                srcs, src_lens, tgt, targets = srcs.to(self.device), \
                    src_lens.to(self.device), tgt.to(self.device), \
                    targets.to(self.device)

                logit = self.model(srcs, src_lens, tgt)
                val_loss = self.cal_loss(logit, targets)
                val_total_loss += val_loss

            avg_loss = val_total_loss / len(self.val_loader)
            print("Epoch {}, validation average loss: {:.4f}".format(
                self.epoch, avg_loss
            ))
            self.current_val_loss = avg_loss
        return avg_loss

    def checkpoint(self):
        save_dict = {}
        name = 'ckpt-{:6f}-{}e-{}s'.format(
            self.current_val_loss, self.epoch, self.step
        )
        save_dict['state_dict'] = self.model.state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()
        torch.save(save_dict, join(self.save_dir, name))

    def log_info(self, losses):
        total_step = len(self.train_loader)
        print("Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}".format(
            self.epoch, self.step, total_step,
            100 * self.step / total_step,
            losses / self.print_freq
            ))

    def check_stop(self, val_loss):
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.checkpoint()
            self.current_p = 0
        else:
            self.current_p += 1
        return self.current_p >= self.patience

    def train(self):

        while True:
            self.model.train()
            losses = 0.0
            for srcs, targets in self.train_loader:
                step_loss = self.train_step(srcs, targets)
                losses += step_loss

                if self.step % self.print_freq == 0:
                    #log message
                    self.log_info(losses)
                    losses = 0.0
                if self.step % self.ckpt_freq == 0:
                    #save current model
                    self.checkpoint()


            self.epoch += 1
            self.step = 0
            # get val loss and
            # check whether to early stop
            val_loss = self.validate()
            if self.check_stop(val_loss):
                print("Finished Training!")
                self.checkpoint()
                break

    def train_step(self, srcs, targets):
        self.optimizer.zero_grad()
        src, src_lens, tgt, extend_src, ext_vsize = srcs
        src, src_lens, tgt, extend_src, targets = src.to(self.device), \
            src_lens.to(self.device), tgt.to(self.device), \
            extend_src.to(self.device), targets.to(self.device)
        #return logit: [batch_size, max_len, voc_size]
        if self.model_is_copy:
            logit = self.model(src, src_lens, tgt,
                               extend_src, ext_vsize).to(self.device)
        else:
            logit = self.model(src, src_lens, tgt).to(self.device)

        loss = self.cal_loss(logit, targets)
        self.step += 1

        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()

    def cal_loss(self, logits, targets, pad_idx=0):
        #logits: [batch_size, max_len, voc_size]
        #targets: [batch_size, max_tgt_len]

        mask = (targets != pad_idx)
        targets = targets.masked_select(mask)
        logits = logits.masked_select(
            mask.unsqueeze(2).expand_as(logits)
        ).contiguous().view(-1, logits.size(2))

        #import pdb;pdb.set_trace()
        loss = F.nll_loss(logits, targets).to(self.device)
        return loss
