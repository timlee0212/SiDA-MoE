import os, shutil
import sys
import torch
import torchvision
import torch.nn as nn
import numpy as np
import random
import csv
import time
from concurrent.futures.thread import ThreadPoolExecutor

import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim

# import imageio
from scipy.stats import truncnorm

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))

# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 80 #int(term_width)
TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
 
class Helper:
  #All directories are end with /
  
  @staticmethod
  def get_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res
  
  @staticmethod
  def pairwise_L2(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
  
  @staticmethod
  def write2txt(anything, file_path):
      with open(file_path, 'w') as f:
          print(anything, file=f) 

  @staticmethod
  def network_norm(Module):
      norm = 0.0
      counter = 0.0
      for name, param in Module.named_parameters():
          if 'weight' in name:
              counter += 1
              norm += param.cpu().clone().detach().norm()/torch.sum(torch.ones(param.shape))
          elif 'bias' in name:
              counter += 1
              norm += param.cpu().clone().detach().norm()/torch.sum(torch.ones(param.shape))
      return (norm/counter).item()
   
  ###======================== Systems ======================== ####
  @staticmethod
  def multithread(max_workers, func, *args):  
      with ThreadPoolExecutor(max_workers=20) as executor:
          func(args)
          
  ###======================== Utilities ====================== ####
  @staticmethod
  def add_common_used_parser(parser):
    #=== directories ===
    parser.add_argument('--exp_name', type=str, default='Test', help='The name for different experimental runs.')
    parser.add_argument('--exp_dir', type=str, default='../experiments/', help='Locations to save different experimental runs.')
    
    #== plot figures ===
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    return parser
     
  @staticmethod
  def get_output_dir(exp_dir, exp_name):
    exp_dir = os.path.join(exp_dir, exp_name)
    output_dir = dict()
    output_dir['codes']  = os.path.join(exp_dir, 'codes/')
    output_dir['checkpoints']  = os.path.join(exp_dir, 'checkpoints/')
    output_dir['logs']  = os.path.join(exp_dir, 'logs/')
    output_dir['figures']  = os.path.join(exp_dir, 'figures/')
    output_dir['results']  = os.path.join(exp_dir, 'results/')
    for name, _dir in output_dir.items():
      if not os.path.isdir(_dir):
        print('Create {} directory: {}'.format(name, _dir))
        os.makedirs(_dir)
    return output_dir

  @staticmethod  
  def backup_codes(src_d, tgt_d, save_types=['.py', '.txt', '.sh', '.out']):
    for root, dirs, files in os.walk(src_d):
      for filename in files:
        type_list = [filename.endswith(tp) for tp in save_types]
        if sum(type_list):
          file_path = os.path.join(root, filename)
          tgt_dir   = root.replace(src_d, tgt_d)
          if not os.path.isdir(tgt_dir):
            os.makedirs(tgt_dir)
          shutil.copyfile(os.path.join(root, filename), os.path.join(tgt_dir, filename))
      
  @staticmethod
  def try_make_dir(d):
    if not os.path.isdir(d):
      # os.mkdir(d)
      os.makedirs(d) # nested is allowed

  @staticmethod
  def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

  @staticmethod
  def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    #To let the cuDNN use the same convolution every time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  @staticmethod
  def create_optimizer(model_parameters, optimizer_type, **optimizer_params):
      if optimizer_type == 'sgd':
          return optim.SGD(model_parameters, lr=optimizer_params['lr'], momentum=optimizer_params.get('momentum', 0.9), weight_decay=optimizer_params.get('weight_decay', 0), nesterov=optimizer_params.get('nesterov', False))

      elif optimizer_type == 'adam':
          return optim.Adam(model_parameters, lr=optimizer_params['lr'], betas=optimizer_params.get('betas', (0.9, 0.999)), eps=optimizer_params.get('eps', 1e-08), weight_decay=optimizer_params.get('weight_decay', 0), amsgrad=optimizer_params.get('amsgrad', False))

      elif optimizer_type == 'adamw':
          return optim.AdamW(model_parameters, lr=optimizer_params['lr'], betas=optimizer_params.get('betas', (0.9, 0.999)), eps=optimizer_params.get('eps', 1e-08), weight_decay=optimizer_params.get('weight_decay', 0), amsgrad=optimizer_params.get('amsgrad', False))

      elif optimizer_type == 'rmsprop':
          return optim.RMSprop(model_parameters, lr=optimizer_params['lr'], alpha=optimizer_params.get('alpha', 0.99), eps=optimizer_params.get('eps', 1e-08), weight_decay=optimizer_params.get('weight_decay', 0), momentum=optimizer_params.get('momentum', 0), centered=optimizer_params.get('centered', False))

      else:
          raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

  @staticmethod
  def create_scheduler(optimizer, scheduler_type, **scheduler_params):
      if scheduler_type == 'step':
          return lr_scheduler.StepLR(optimizer, step_size=scheduler_params['step_size'], gamma=scheduler_params['gamma'])

      elif scheduler_type == 'multi_step':
          return lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_params['milestones'], gamma=scheduler_params['gamma'])

      elif scheduler_type == 'exponential':
          return lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_params['gamma'])

      elif scheduler_type == 'cosine_annealing':
          return lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_params['T_max'], eta_min=scheduler_params['eta_min'])

      elif scheduler_type == 'cosine_annealing_warm_restarts':
          return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=scheduler_params['T_0'], T_mult=scheduler_params['T_mult'], eta_min=scheduler_params['eta_min'])

      elif scheduler_type == 'cyclic':
          return lr_scheduler.CyclicLR(optimizer, base_lr=scheduler_params['base_lr'], max_lr=scheduler_params['max_lr'], step_size_up=scheduler_params['step_size_up'], step_size_down=scheduler_params['step_size_down'], mode=scheduler_params['mode'], gamma=scheduler_params['gamma'])

      elif scheduler_type == 'one_cycle':
          return lr_scheduler.OneCycleLR(optimizer, max_lr=scheduler_params['max_lr'], total_steps=scheduler_params['total_steps'], epochs=scheduler_params['epochs'], steps_per_epoch=scheduler_params['steps_per_epoch'], pct_start=scheduler_params['pct_start'], anneal_strategy=scheduler_params['anneal_strategy'], div_factor=scheduler_params['div_factor'], final_div_factor=scheduler_params['final_div_factor'])

      else:
          raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

  ###======================== Logs ======================== ####
  @staticmethod
  def log(logf, msg, mode='a', console_print=True):
    with open(logf, mode) as f:
        f.write(msg + '\n')
    if console_print:
        print(msg)
     
     
  @staticmethod
  def write_dict2csv(log_dir, write_dict, mode="a"):
    for key in write_dict.keys():
      with open(log_dir + key + '.csv', mode) as f:
        if isinstance(write_dict[key], str):
          f.write(write_dict[key])
        elif isinstance(write_dict[key], list):
          writer = csv.writer(f)
          writer.writerow(write_dict[key])
        else:
          raise ValueError("write_dict has wrong type")
  
  @staticmethod
  def progress_bar(current, total, msg=None):
      global last_time, begin_time
      if current == 0:
          begin_time = time.time()  # Reset for new bar.

      cur_len = int(TOTAL_BAR_LENGTH*current/total)
      rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

      sys.stdout.write(' [')
      for i in range(cur_len):
          sys.stdout.write('=')
      sys.stdout.write('>')
      for i in range(rest_len):
          sys.stdout.write('.')
      sys.stdout.write(']')

      cur_time = time.time()
      step_time = cur_time - last_time
      last_time = cur_time
      tot_time = cur_time - begin_time

      L = []
      L.append('  Step: %s' % format_time(step_time))
      L.append(' | Tot: %s' % format_time(tot_time))
      if msg:
          L.append(' | ' + msg)

      msg = ''.join(L)
      sys.stdout.write(msg)
      for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
          sys.stdout.write(' ')

      # Go back to the center of the bar.
      for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
          sys.stdout.write('\b')
      sys.stdout.write(' %d/%d ' % (current+1, total))

      if current < total-1:
          print(current, end='\r')
      else:
          sys.stdout.write('\n')
      sys.stdout.flush()
    
  @staticmethod
  def is_base_net(module):
      if isinstance(module, nn.Linear):
          return True
      if isinstance(module, nn.Conv2d):
          return True
      # if isinstance(module, nn.BatchNorm2d):
      #     return True
      
      return False

  
 	###======================== Visualization ================= ###
  @staticmethod
  def save_images(samples, sample_dir, sample_name, offset=0, nrows=0):
    if nrows == 0:
      bs = samples.shape[0]
      nrows = int(bs**.5)
    if offset > 0:
      sample_name += '_' + str(offset)
    save_path = os.path.join(sample_dir, sample_name + '.png')
    torchvision.utils.save_image(samples.cpu(), save_path, nrow=nrows, normalize=True) 