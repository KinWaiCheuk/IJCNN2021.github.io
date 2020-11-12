import os


from datetime import datetime
import pickle

import numpy as np
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable


from evaluate import evaluate_wo_velocity
from model import *

import matplotlib.pyplot as plt
ex = Experiment('train_original')

# parameters for the network
ds_ksize, ds_stride = (2,2),(2,2)
mode = 'imagewise'
sparsity = 1
output_channel = 2
logging_freq = 500
saving_freq = 1000


@ex.config
def config():
    # logdir = f'runs_AE/test' + '-' + datetime.now().strftime('%y%m%d-%H%M%S')
    # Choosing GPU to use
#     GPU = '0'
#     os.environ['CUDA_VISIBLE_DEVICES']=str(GPU)
    device = 'cuda:0'
    log = True
    w_size = 30
    spec = 'Mel'
    resume_iteration = None
    train_on = 'MAPS'
    attention=True
    layers=2
    cat_feat = True
    onset=False
    
    batch_size = 16
    sequence_length = 327680
    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    epoches = 20000        
    learning_rate = 6e-5
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98
    label_type = 'onset'

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    refresh = False
    if attention==True:
        logdir = f'runs/Simple_{layers}l_cat_feat_{cat_feat}-attent-{w_size}-onset_{onset}-{train_on}-{spec}-{mode}_log-{log}_lr-{learning_rate}-'+ datetime.now().strftime('%y%m%d-%H%M%S')
    else:
        logdir = f'runs/Simple-{layers}l_{train_on}-{spec}-{mode}_log-{log}_lr-{learning_rate}-'+ datetime.now().strftime('%y%m%d-%H%M%S')        
        
    ex.observers.append(FileStorageObserver.create(logdir)) # saving source code
def flatten_attention(a, w_size=30):
#     seq_len = a.shape[1]
    seq_len = a.shape[0]
    attentions = torch.zeros(seq_len, 1 , seq_len)
    for t in range(seq_len):
        start = 0 if t-w_size<0 else t-w_size
        end = seq_len if t+w_size > seq_len else t+w_size
#         print(f'start={start}\tend={end}\t{attentions[t, :, start:end].shape}')
        if t<w_size:
            attentions[t, :, start:end+1] = a[t, :, -(end-start)-1:]
        else:
            attentions[t, :, start:end] = a[t, :, :(end-start)]
            
    return attentions        
@ex.automain
def train(spec, resume_iteration, train_on, batch_size, sequence_length,w_size, attention, layers, cat_feat,
          learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out, label_type,
          clip_gradient_norm, validation_length, refresh, device, epoches, logdir, log, onset): 
    print_config(ex.current_run)

        
    train_groups, validation_groups = ['train'], ['validation'] # Parameters for MAESTRO

    if leave_one_out is not None: # It applies only to MAESTRO
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    # Choosing the dataset to use
    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length, device=device)
        # validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length, device=device, refresh=refresh)

    elif train_on == 'MusicNet':
        dataset = MusicNet(groups=['train'], sequence_length=sequence_length, device=device, refresh=refresh)
        validation_dataset = MusicNet(groups=['test'], sequence_length=sequence_length, device=device, refresh=refresh)

    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length, overlap=False, device=device, refresh=refresh)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length, overlap=True, device=device, refresh=refresh)
        
    full_validation = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=None, device=device, refresh=refresh)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(validation_dataset, 4, shuffle=False, drop_last=True)
    batch_visualize = next(iter(valloader)) # Getting one fixed batch for visualization

    if resume_iteration is None:
        
        model = SimpleModel(N_BINS, MAX_MIDI - MIN_MIDI + 1, log=log, mode=mode, spec=spec, device=device, w_size=w_size, attention=attention, layers=layers, cat_feat=cat_feat, onset=onset)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else: # Loading checkpoints and continue training
        trained_dir='trained_MAPS' # Assume that the checkpoint is in this folder
        model_path = os.path.join(trained_dir, f'{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(trained_dir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    # loop = tqdm(range(resume_iteration + 1, iterations + 1))
    
    total_batch = len(loader.dataset)
    for ep in range(1, epoches+1):
        model.train()
        total_loss = 0
        batch_idx = 0
        # print(f'ep = {ep}, lr = {scheduler.get_lr()}')
        for batch in loader:
            predictions, losses, _ = model.run_on_batch(batch)

            loss = sum(losses.values())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if clip_gradient_norm:
                clip_grad_norm_(model.parameters(), clip_gradient_norm)
            batch_idx += 1
            print(f'Train Epoch: {ep} [{batch_idx*batch_size}/{total_batch}'
                    f'({100. * batch_idx*batch_size / total_batch:.0f}%)]'
                    f'\tLoss: {loss.item():.6f}'
                    , end='\r') 
        print(' '*100, end = '\r')          
        print(f'Train Epoch: {ep}\tLoss: {total_loss/len(loader):.6f}')
        
        
        # Logging results to tensorboard
        if ep == 1:
#             os.makedirs(logdir, exist_ok=True) # creating the log dir
            writer = SummaryWriter(logdir) # create tensorboard logger
            
        if (ep)%logging_freq == 0 and ep > 100:
            model.eval()
            with torch.no_grad():
                for key, values in evaluate_wo_velocity(validation_dataset, model, reconstruction=False).items():
                    if key.startswith('metric/'):
                        _, category, name = key.split('/')
                        print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
                        if ('precision' in name or 'recall' in name or 'f1' in name) and 'chroma' not in name:
                            writer.add_scalar(key, np.mean(values), global_step=ep)

        if (ep)%saving_freq == 0:
            torch.save(model.state_dict(), os.path.join(logdir, f'model-{ep}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=ep)

        # Load one batch from validation_dataset
            
        predictions, losses, mel = model.run_on_batch(batch_visualize)


        if ep==1: # Showing the original transcription and spectrograms
            fig, axs = plt.subplots(2, 2, figsize=(24,8))
            axs = axs.flat
            for idx, i in enumerate(mel.cpu().detach().numpy()):
                axs[idx].imshow(i.transpose(), cmap='jet', origin='lower')
                axs[idx].axis('off')
            fig.tight_layout()

            writer.add_figure('images/Original', fig , ep)

            fig, axs = plt.subplots(2, 2, figsize=(24,4))
            axs = axs.flat
            for idx, i in enumerate(batch_visualize['frame'].cpu().numpy()):
                axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
                axs[idx].axis('off')
            fig.tight_layout()
            writer.add_figure('images/Label', fig , ep)

        if ep%logging_freq == 0:
            fig, axs = plt.subplots(2, 2, figsize=(24,4))
            axs = axs.flat
            for idx, i in enumerate(predictions['frame'].detach().cpu().numpy()):
                axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
                axs[idx].axis('off')
            fig.tight_layout()
            writer.add_figure('images/Transcription', fig , ep)
            if attention==True:         
                # show attention    
                fig, axCenters = plt.subplots(2,2, figsize=(24, 24))
                fig.subplots_adjust(.05,.1,.95,.95)
                for idx, axCenter in enumerate(axCenters.flat):
                    attentions = flatten_attention(predictions['attention'][idx],w_size=w_size)
                    attended_features = mel[idx].detach().cpu().squeeze().t()

                    divider = make_axes_locatable(axCenter)
                    axvert = divider.append_axes('left', size='30%', pad=0.5)
                    axhoriz = divider.append_axes('top', size='20%', pad=0.25)
                    axCenter.imshow(attentions.squeeze(1).detach().cpu(), cmap='jet', vmin=0, vmax=0.7)
                    #     if attended_features=='activation':
                    #         attended = pred[attended_features][0].t().detach().cpu()
                    #     else:
                    #         attended = pred[attended_features].t().detach().cpu()
                    axhoriz.imshow(attended_features, aspect='auto', origin='lower', cmap='jet')
                    axvert.imshow(predictions['frame'][idx].detach().cpu(), aspect='auto')

                    # changing axis for the center fig
                    axCenter.set_xticks([])

                    # changing axis for the output fig (left fig)
                    axvert.set_yticks([])
                    axvert.xaxis.tick_top()
                    axvert.set_title('Transcription')

                    axhoriz.set_title(f'Attended Feature (Spec)')

                    axhoriz.margins(x=0)
                    axvert.margins(y=0)
                writer.add_figure('images/Attention', fig , ep)


           
                
    # Evaluating model performance on the full MAPS songs in the test split     
    print('Training finished, now evaluating on the MAPS test split (full songs)')
    validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=None, device=device, refresh=refresh)
    with torch.no_grad():
        model = model.eval()
        metrics = evaluate_wo_velocity(tqdm(validation_dataset), model, reconstruction=False,
                                       save_path=os.path.join(logdir,'./MIDI_results'))
        
    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
         
    export_path = os.path.join(logdir, 'result_dict')    
    pickle.dump(metrics, open(export_path, 'wb'))
