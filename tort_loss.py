import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TortLoss(nn.Module):
    def __init__(self, use_rep_loss, use_sl_loss, use_rot_loss, device, rep_w, sl_w, rot_w,
                 out_dim, global_crops_number, masked_crops_number, local_crops_number, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp_epochs, num_epochs, student_temp, 
                 center_momentum, smoothing):
        super().__init__()
        assert use_rep_loss or use_sl_loss or use_rot_loss, 'At least one loss function must be used.'
        self.use_rep_loss = use_rep_loss
        self.use_sl_loss = use_sl_loss
        self.use_rot_loss = use_rot_loss
        if self.use_rep_loss:
            
            self.rep_loss_fn = RepresentationCrossEntropyLoss(out_dim, 
                                global_crops_number, masked_crops_number, local_crops_number, 
                                warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, 
                                num_epochs, student_temp, center_momentum).to(device)
        if self.use_sl_loss:
            self.sl_loss_fn = CrossEntropyLoss(smoothing).to(device)
        if self.use_rot_loss:
            self.rot_loss_fn = CrossEntropyLoss().to(device)
        
        self.rep_w = rep_w
        self.sl_w = sl_w
        self.rot_w = rot_w
    
    def forward(self, student_output, teacher_output, epoch, 
                sl_pred, sl_labels, 
                rot_pred, rot_labels):
        rep_loss, sl_loss, rot_loss = 0, 0, 0
        if self.use_rep_loss:
            rep_loss = self.rep_loss_fn(student_output, teacher_output, epoch)
        if self.use_sl_loss:
            sl_loss = self.sl_loss_fn(sl_pred, sl_labels)
        if self.use_rot_loss:
            rot_loss = self.rot_loss_fn(rot_pred, rot_labels)
        loss = rep_loss * self.rep_w + sl_loss * self.sl_w + rot_loss * self.rot_w
        return loss, rep_loss, sl_loss, rot_loss


class RepresentationCrossEntropyLoss(nn.Module):
    def __init__(self, out_dim, n_g_crops, n_m_crops, n_add_crops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_gm_crops = n_g_crops + n_m_crops
        self.n_crops = n_g_crops + n_m_crops + n_add_crops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.n_crops)

        # teacher centering and sharpening
        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.n_gm_crops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class CrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=None):
        super().__init__()
        if smoothing is not None:
            from timm.loss import LabelSmoothingCrossEntropy
            self.loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, prediction, label):
        return self.loss(prediction, label)
