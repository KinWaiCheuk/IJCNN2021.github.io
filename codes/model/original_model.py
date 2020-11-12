"""
A rough translation of Magenta's Onsets and Frames implementation [1].
    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn
from nnAudio import Spectrogram
from .constants import *
from model.utils import Normalization



class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, spec):
        x = spec.view(spec.size(0), 1, spec.size(1), spec.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

class Onset_Stack(nn.Module):
    def __init__(self, input_features, model_size, output_features, sequence_model):
        super().__init__()
        self.convstack = ConvStack(input_features, model_size)
        self.sequence_model = sequence_model
        if self.sequence_model:
            self.linear = nn.Linear(model_size, output_features)
            self.forward = self.forward_LSTM
        else:
            self.linear = nn.Linear(model_size, output_features)
            self.forward = self.forward_noLSTM
        
    def forward_LSTM(self, x):
        x = self.convstack(x)
        x, (h, c) = self.sequence_model(x)
        x = self.linear(x)
        
        return torch.sigmoid(x)
    
    def forward_noLSTM(self, x):
        x = self.convstack(x)
        x = self.linear(x)
        
        return torch.sigmoid(x)        
    
    
class Combine_Stack(nn.Module):
    def __init__(self, model_size, output_features, sequence_model):
        super().__init__()
        self.sequence_model = sequence_model
        if self.sequence_model:
            self.linear = nn.Linear(model_size, output_features)
        else:
            self.linear = nn.Linear(output_features*2, output_features)        
        
        if sequence_model:
            self.forward = self.forward_LSTM
        else:
            self.forward = self.forward_noLSTM
        
    def forward_LSTM(self, x):
        x, (h,c) = self.sequence_model(x)
        x = self.linear(x)
        
        return torch.sigmoid(x)

    def forward_noLSTM(self,x):
        x = self.linear(x)
        return torch.sigmoid(x)

    
class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, log=True, mode='imagewise', spec='Mel', onset_stack=True, LSTM=True):
        super().__init__()
        self.onset_stack=onset_stack
        self.log = log
        self.normalize = Normalization(mode)        
        
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: nn.LSTM(input_size, output_size // 2,  batch_first=True, bidirectional=True)
        # Need to rewrite this part, since we are going to modify the LSTM
        print
        if onset_stack==True:
            if LSTM==True:
                self.onset_stack = Onset_Stack(input_features, model_size, output_features, sequence_model(model_size, model_size))
                self.combined_stack = Combine_Stack(model_size, output_features, sequence_model(output_features * 2, model_size))
            else:
                self.onset_stack = Onset_Stack(input_features, model_size, output_features, None)
                self.combined_stack = Combine_Stack(model_size, output_features, None)                
        else:
            if LSTM==True:
                self.combined_stack = Combine_Stack(model_size, output_features, sequence_model(output_features, model_size))
            else:
                self.combined_stack = Combine_Stack(model_size, output_features, None)


        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        
        if self.onset_stack: 
            print('I still have onset')
            self.forward = self.forward_onset_stack
        else:
            self.forward = self.forward_frame_only        

    def forward_onset_stack(self, spec):
        onset_pred = self.onset_stack(spec)
#         offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(spec)
        combined_pred = torch.cat([onset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
#         velocity_pred = self.velocity_stack(mel)
        
        return onset_pred, activation_pred, frame_pred
    
    def forward_frame_only(self, spec):
        activation_pred = self.frame_stack(spec)    
        frame_pred = self.combined_stack(activation_pred) # Attenting on onset     
        return None, activation_pred, frame_pred    

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        if self.onset_stack:
            onset_label = batch['onset']
#         offset_label = batch['offset']
        frame_label = batch['frame']
#         velocity_label = batch['velocity']
  
        spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
        
        onset_pred, _, frame_pred = self(spec)
        if self.onset_stack:
            predictions = {
                'onset': onset_pred.reshape(*onset_label.shape),
    #             'offset': offset_pred.reshape(*offset_label.shape),
                'frame': frame_pred.reshape(*frame_label.shape),
    #             'velocity': velocity_pred.reshape(*velocity_label.shape)
            }

            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
    #             'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
    #             'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
        else:
            predictions = {
                'frame': frame_pred.reshape(*frame_label.shape),
                'onset': frame_pred.reshape(*frame_pred.shape)
            }

            losses = {
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            }        

        return predictions, losses, spec
    
    def load_my_state_dict(self, state_dict):
        """Useful when loading part of the weights. From https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2"""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


