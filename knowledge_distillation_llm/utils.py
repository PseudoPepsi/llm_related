"""几种kl div的计算方法"""
import torch

# 计算前向kl散度
def compute_fkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id,
        reduction="sum",
        temp = 1.0,  # temperature
    ):
        logits = logits / temp # student logits q(x)
        teacher_logits = teacher_logits / temp # teacher logits q(x)

        log_probs = torch.log_softmax(logits, -1, dtype=torch.float32) # student prob q(x)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32) # teacher prob p(x)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32) # log(p(x))
        kl = (teacher_probs * (teacher_log_probs - log_probs)) # p(x)log(p(x)/q(x))
        kl = kl.sum(-1) # integration
        if reduction == "sum": # 只需要输出的部分，不需要输入和pad填充部分
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            kl = kl.sum()

        return kl
# 计算反向kl散度
def compute_rkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id,
        reduction="sum", 
        temp = 1.0
    ):
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (probs * (log_probs - teacher_log_probs))
        kl = kl.sum(-1)
        if reduction == "sum":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            kl = kl.sum()
        return kl

# 计算偏向前kl散度
def compute_skewed_fkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id, 
        reduction="sum", 
        temp = 1.0,
        skew_lambda = 0.1
    ):
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = skew_lambda * teacher_probs + (1 - skew_lambda) * probs
        mixed_log_probs = torch.log(mixed_probs)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (teacher_probs * (teacher_log_probs - mixed_log_probs))
        kl = kl.sum(-1)
        if reduction == "sum":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            kl = kl.sum()

            
        return kl
# 计算偏向反kl散度    
def compute_skewed_rkl(
    logits, 
    teacher_logits, 
    target,
    padding_id,
    reduction="sum", 
    temp = 1.0,
    skew_lambda = 0.1
):
    logits = logits / temp
    teacher_logits = teacher_logits / temp
    
    probs = torch.softmax(logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * probs
    mixed_log_probs = torch.log(mixed_probs)
    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    kl = (probs * (log_probs - mixed_log_probs))
    kl = kl.sum(-1)
    
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        kl = kl.sum()


    return kl