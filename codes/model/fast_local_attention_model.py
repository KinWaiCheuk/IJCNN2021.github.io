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

class Attention(nn.Module):
    def __init__(self, input_dim, output_dim, model_size):
        super().__init__()
        
        self.attn = nn.Linear(input_dim + model_size, output_dim)
        self.v = nn.Linear(output_dim, 1, bias=False)
        
        self.input_dim = input_dim + model_size
       
    def forward(self, hidden, encoder_outputs):
        # hidden should be of the shape [batch size, hid dim]
        batch_size = encoder_outputs.shape[0]
        encoder_outputs = encoder_outputs
        src_len = encoder_outputs.shape[2]
        hidden = hidden.unsqueeze(2).repeat(1, 1, src_len, 1)
        # hidden =          (batch, spec_len, local_att_len, features)
        # encoder_outputs = (batch, spec_len, local_att_len, features)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=-1)))
        attention = self.v(energy).squeeze(-1)
        # a shape = (torch.Size([1, 640, 1, 61])) 
        return torch.softmax(attention, -1)
        
        
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
                    
        
        self.linear = nn.Linear(model_size, output_features)
    
    def forward_LSTM(self, x):
        x = self.convstack(x)
        x, (h, c) = self.sequence_model(x)
        x = self.linear(x)
        
        return torch.sigmoid(x)
    
    def forward_noLSTM(self, x):
        x = self.convstack(x)
        x = self.linear(x)
        
        return torch.sigmoid(x)      
    
class Combine_Stack_with_attn(nn.Module):
    def __init__(self, model_size, output_features, sequence_model, attention_mode, w_size):
        super().__init__()
        self.sequence_model = sequence_model
        self.w_size = w_size
        
        if self.sequence_model:
            if attention_mode=='spec':
                self.attention = Attention(N_BINS, output_features, model_size)    
                self.linear = nn.Linear(model_size + N_BINS, output_features)
            else:
                self.attention = Attention(output_features, output_features, model_size)
                self.linear = nn.Linear(model_size + output_features, output_features)  
            self.forward = self.forward_LSTM
        else:
            if attention_mode=='spec':
                self.attention = Attention(N_BINS, output_features, output_features * 2)    
                self.linear = nn.Linear(output_features * 2 + N_BINS, output_features)
            else:
                self.attention = Attention(output_features, output_features, output_features * 2)
                self.linear = nn.Linear(output_features * 2 + output_features, output_features) 
            self.forward = self.forward_noLSTM
        
    def forward_LSTM(self, x, encoder_outputs):
        x, _ = self.sequence_model(x)
        
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        feature_size = encoder_outputs.shape[2]   
        # encoder_outputs.shape = (batch, seq_len, features)
        encoder_outputs = nn.functional.pad(encoder_outputs,(0,0,self.w_size,self.w_size)) #Padding the sequence so that the local attention size is fixed and can be packed into a tensor
        padded_seq_len = encoder_outputs.shape[1]  
        windowed_encoder_outputs = torch.as_strided(encoder_outputs, 
                                                    (batch_size, seq_len, self.w_size*2+1, feature_size), 
                                                    (padded_seq_len*feature_size, feature_size, feature_size, 1))
        
        a = self.attention(x, windowed_encoder_outputs)
        a = a.unsqueeze(2)
        # a =                (batch, seq_len, 1, att_len)
        # windowed_outputs = (batch, seq_len, att_len, features)
        weighted = torch.matmul(a, windowed_encoder_outputs).squeeze(2) # remove the extra dim       
        x = self.linear(torch.cat((x, weighted), dim=-1))
        
        return torch.sigmoid(x), a
    
    def forward_noLSTM(self, x, encoder_outputs):       
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        feature_size = encoder_outputs.shape[2]   
        # encoder_outputs.shape = (batch, seq_len, features)
        encoder_outputs = nn.functional.pad(encoder_outputs,(0,0,self.w_size,self.w_size)) #Padding the sequence so that the local attention size is fixed and can be packed into a tensor
        padded_seq_len = encoder_outputs.shape[1]  
        windowed_encoder_outputs = torch.as_strided(encoder_outputs, 
                                                    (batch_size, seq_len, self.w_size*2+1, feature_size), 
                                                    (padded_seq_len*feature_size, feature_size, feature_size, 1))
        
        a = self.attention(x, windowed_encoder_outputs)
        a = a.unsqueeze(2)
        # a =                (batch, seq_len, 1, att_len)
        # windowed_outputs = (batch, seq_len, att_len, features)
        weighted = torch.matmul(a, windowed_encoder_outputs).squeeze(2) # remove the extra dim       
        x = self.linear(torch.cat((x, weighted), dim=-1))
        
        return torch.sigmoid(x), a    
    
class OnsetsAndFrames_with_fast_local_attn(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, log=True, mode='imagewise', spec='Mel', device='cpu', attention_mode='activation', w_size=30, onset_stack=True, LSTM=True):
        super().__init__()
        self.onset_stack=onset_stack
        self.w_size=w_size
        self.device = device
        self.log = log
        self.normalize = Normalization(mode)        
        self.attention_mode=attention_mode
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: nn.LSTM(input_size, output_size // 2,  batch_first=True, bidirectional=True)
        # Need to rewrite this part, since we are going to modify the LSTM
        
        
        
#         self.offset_stack = Offset_Stack(input_features, model_size, output_features, sequence_model(model_size, model_size))
        if onset_stack==True:
            if LSTM==True:            
                self.combined_stack = Combine_Stack_with_attn(model_size, output_features,
                                                              sequence_model(output_features * 2,
                                                                             model_size),
                                                              attention_mode,
                                                              w_size)
                self.onset_stack = Onset_Stack(input_features, model_size, output_features
                                               , sequence_model(model_size, model_size))
            else:
                self.combined_stack = Combine_Stack_with_attn(model_size, output_features,
                                                              None,
                                                              attention_mode,
                                                              w_size)
                self.onset_stack = Onset_Stack(input_features, model_size, output_features
                                               , None)             
             
        else:
            if LSTM==True:
            
                self.combined_stack = Combine_Stack_with_attn(model_size, output_features,
                                                              sequence_model(output_features,
                                                                             model_size),
                                                              attention_mode, w_size)
            else:
                self.combined_stack = Combine_Stack_with_attn(model_size, output_features,
                                                              None,
                                                              attention_mode, w_size)   

    
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        
        if self.onset_stack: 
            self.forward = self.forward_onset_stack
        else:
            self.forward = self.forward_frame_only
    
    def forward_onset_stack(self, spec):
        onset_pred = self.onset_stack(spec)
        seq_len = onset_pred.shape[1]
#         offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(spec)

        combined_pred = torch.cat([onset_pred.detach(), activation_pred], dim=-1)

#         hidden = (h,c) # Setting the first hidden state to be the output from onset stack
        if self.attention_mode=='onset':
            frame_pred, a = self.combined_stack(combined_pred, onset_pred) # Attenting on onset
        elif self.attention_mode=='activation':
            frame_pred, a = self.combined_stack(combined_pred, activation_pred) # Attenting on features
        elif self.attention_mode=='spec':
            frame_pred, a = self.combined_stack(combined_pred, spec) # Attenting on features
        else:
            raise NameError(f"attention_mode={self.attention_mode} is not defined")

        return onset_pred, activation_pred, frame_pred, a
        
    def forward_frame_only(self, spec):
        activation_pred = self.frame_stack(spec)    

        if self.attention_mode=='onset':
            frame_pred, a = self.combined_stack(activation_pred, onset_pred) # Attenting on onset
        elif self.attention_mode=='activation':
            frame_pred, a = self.combined_stack(activation_pred, activation_pred) # Attenting on features
        elif self.attention_mode=='spec':
            frame_pred, a = self.combined_stack(activation_pred, spec) # Attenting on features
        else:
            raise NameError(f"attention_mode={self.attention_mode} is not defined")            

        return None, activation_pred, frame_pred, a 

       
        
    def run_on_batch(self, batch):
        audio_label = batch['audio']
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
        
        
        onset_pred, activation_pred, frame_pred, a = self(spec)
        
        if self.onset_stack:
            predictions = {
#                 'onset': onset_pred.reshape(*onset_label.shape),
                'onset': onset_pred.reshape(*onset_label.shape),
    #             'offset': offset_pred.reshape(*offset_label.shape),
                'activation': activation_pred,
                'frame': frame_pred.reshape(*frame_label.shape),
                'attention': a
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
                'onset': frame_pred.reshape(*frame_label.shape),
                'activation': activation_pred,
                'frame': frame_pred.reshape(*frame_label.shape),
                'attention': a
            }
            losses = {
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            }            

        return predictions, losses, spec
    
    def feed_audio(self, audio):
#         velocity_label = batch['velocity']
  
        spec = self.spectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
        if self.log:
            spec = torch.log(spec + 1e-5)
        # print(f'spec shape = {spec.shape}')
        spec = self.normalize.transform(spec)
        spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
        
        onset_pred, activation_pred, frame_pred, a = self(spec)
        
        predictions = {
            'onset': onset_pred,
#             'offset': offset_pred.reshape(*offset_label.shape),
            'activation': activation_pred,
            'frame': frame_pred,
            'attention': a
#             'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        return predictions, spec

    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)