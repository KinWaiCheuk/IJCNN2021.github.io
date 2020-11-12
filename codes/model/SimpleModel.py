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
    def __init__(self, s_dim, model_size, H_dim):
        super().__init__()
        
        self.attn = nn.Linear(s_dim + H_dim, model_size)
        self.v = nn.Linear(model_size, 1, bias=False)
        
       
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

class Frame_Stack(nn.Module):
    def __init__(self, model_size, output_features, layers=2):
        super().__init__()
        if layers==1:
            self.linear1 = nn.Linear(N_BINS, output_features)
            self.forward_linear = self.forward_linear1
        elif layers==2:
            self.linear1 = nn.Linear(N_BINS, model_size)
            self.linear2 = nn.Linear(model_size, output_features)  
            self.forward_linear = self.forward_linear2      

    def forward_linear1(self, x):
        x = self.linear1(x)
        return x               
            
    def forward_linear2(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x     
 
    def forward(self, x):
        x = self.forward_linear(x)
        return torch.sigmoid(x)      
    
class Frame_Stack_with_attn(nn.Module):
    def __init__(self, model_size, output_features, w_size, layers=2, cat_feat=True):
        super().__init__()
        self.w_size = w_size
        
        
        if layers==1:
            self.attention = Attention(N_BINS, model_size, N_BINS)    
            if cat_feat==True:
                self.linear1 = nn.Linear(N_BINS+N_BINS, output_features)
                self.forward_linear = self.forward_linear1_cat_feat
            else:    
                self.linear1 = nn.Linear(N_BINS, output_features)
                self.forward_linear = self.forward_linear1_feat
        elif layers==2:
            self.attention = Attention(N_BINS, model_size, N_BINS)    
            self.linear1 = nn.Linear(N_BINS, model_size)
            if cat_feat==True:
                self.linear2 = nn.Linear(model_size+N_BINS, output_features)
                self.forward_linear = self.forward_linear2
            else:
                raise Exception("cat_feat can't be used under layers=2") 
        else:
            print('Error: Layers can be only 1 or 2')

    def forward_linear1_cat_feat(self, x, weighted):
        x = self.linear1(torch.cat((x, weighted), dim=-1))
        return x
    
    def forward_linear1_feat(self, x, weighted):
        x = self.linear1(weighted)
        return x    
    
    def forward_linear2(self, x, weighted):
        x = torch.relu(self.linear1(x))
        x = self.linear2(torch.cat((x, weighted), dim=-1))
        return x
      
            
    def forward(self, x, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        feature_size = encoder_outputs.shape[2]        
        # encoder_outputs.shape = (batch, seq_len, features)
        
        #Padding the sequence so that the local attention size is fixed and can be packed into a tensor
        encoder_outputs = nn.functional.pad(encoder_outputs,
                                            (0,0,self.w_size,self.w_size)) 
        padded_seq_len = encoder_outputs.shape[1]        
        windowed_encoder_outputs = torch.as_strided(encoder_outputs, 
                                                    (batch_size, seq_len, self.w_size*2+1, feature_size), 
                                                    (padded_seq_len*feature_size, feature_size, feature_size, 1))

        a = self.attention(x, windowed_encoder_outputs)
        a = a.unsqueeze(2)
        # a =                (batch, seq_len, 1, att_len)
        # windowed_outputs = (batch, seq_len, att_len, features)
        weighted = torch.matmul(a, windowed_encoder_outputs).squeeze(2) # remove the extra dim       
        x = self.forward_linear(x, weighted)
        
        return torch.sigmoid(x), a

    
class Onset_Stack(nn.Module):
    def __init__(self, output_features):
        super().__init__()         
        self.linear = nn.Linear(N_BINS, output_features)
            
    def forward(self, x):     
        x = self.linear(x)
        return torch.sigmoid(x)
   
    
class SimpleModel(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=128, log=True, mode='imagewise', spec='Mel', device='cpu', w_size=30, attention=True, layers=1, cat_feat=False, onset=False):
        super().__init__()
        self.layers = layers
        self.w_size=w_size
        self.device = device
        self.log = log
        self.normalize = Normalization(mode)        
        self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
                                                      hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                      trainable_mel=False, trainable_STFT=False)
        
        model_size = model_complexity
        if attention==True:
            if onset==True:
                self.onset_stack = Onset_Stack(output_features)
                self.combined_stack = Frame_Stack_with_attn(model_size, output_features, w_size, layers, cat_feat)
                self.forward = self.forward_w_attention_onset
                self.run_on_batch = self.run_on_batch_w_attention_onset
            else:
                self.combined_stack = Frame_Stack_with_attn(model_size, output_features, w_size, layers, cat_feat)
                self.forward = self.forward_w_attention
                self.run_on_batch = self.run_on_batch_w_attention   
                
        elif attention==False:
            self.combined_stack = Frame_Stack(model_size, output_features, layers)
            self.forward = self.forward_wo_attention
            self.run_on_batch = self.run_on_batch_wo_attention
        
    def forward_w_attention(self, spec):
        frame_pred, a = self.combined_stack(spec, spec) # Attenting on features
        return frame_pred, a 
    
    def forward_w_attention_onset(self, spec):
        frame_pred, a = self.combined_stack(spec, spec) # Attenting on features
        onset_pred = self.onset_stack(spec)
        return frame_pred, a, onset_pred 
    
    def forward_wo_attention(self, spec):
        frame_pred = self.combined_stack(spec) # Attenting on features
        return frame_pred

               
    def run_on_batch_w_attention(self, batch):
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
        frame_pred, a = self(spec)
        predictions = {
            'onset': frame_pred.reshape(*frame_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'attention': a
        }
        losses = {
            'loss/frame': F.binary_cross_entropy(predictions['frame'].squeeze(0), frame_label),
        }           
            
        return predictions, losses, spec
    
    def run_on_batch_w_attention_onset(self, batch):
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
        frame_pred, a, onset_pred = self(spec)
        predictions = {
            'onset': onset_pred.reshape(*onset_pred.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'attention': a
        }
        losses = {
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
        }           
            
        return predictions, losses, spec    
    
    def run_on_batch_wo_attention(self, batch):
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
        
        frame_pred = self(spec)
        predictions = {
            'onset': frame_pred.reshape(*frame_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
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
        
        
        frame_pred, a = self(spec)
        
        predictions = {
#             'offset': offset_pred.reshape(*offset_label.shape),
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


# """
# A rough translation of Magenta's Onsets and Frames implementation [1].
#     [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
# """

# import torch
# import torch.nn.functional as F
# from torch import nn
# from nnAudio import Spectrogram
# from .constants import *
# from model.utils import Normalization

# class Attention(nn.Module):
#     def __init__(self, s_dim, model_size, H_dim):
#         super().__init__()
        
#         self.attn = nn.Linear(s_dim + H_dim, model_size)
#         self.v = nn.Linear(model_size, 1, bias=False)
        
       
#     def forward(self, hidden, encoder_outputs):
#         # hidden should be of the shape [batch size, hid dim]
#         batch_size = encoder_outputs.shape[0]
#         encoder_outputs = encoder_outputs
#         src_len = encoder_outputs.shape[2]
#         hidden = hidden.unsqueeze(2).repeat(1, 1, src_len, 1)
#         # hidden =          (batch, spec_len, local_att_len, features)
#         # encoder_outputs = (batch, spec_len, local_att_len, features)
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=-1)))
#         attention = self.v(energy).squeeze(-1)
#         # a shape = (torch.Size([1, 640, 1, 61])) 
#         return torch.softmax(attention, -1)

# class Combine_Stack(nn.Module):
#     def __init__(self, model_size, output_features, layers=2):
#         super().__init__()
#         if layers==1:
#             self.linear1 = nn.Linear(N_BINS, output_features)
#             self.forward_linear = self.forward_linear1
#         elif layers==2:
#             self.linear1 = nn.Linear(N_BINS, model_size)
#             self.linear2 = nn.Linear(model_size, output_features)  
#             self.forward_linear = self.forward_linear2      

#     def forward_linear1(self, x):
#         x = self.linear1(x)
#         return x               
            
#     def forward_linear2(self, x):
#         x = torch.relu(self.linear1(x))
#         x = self.linear2(x)
#         return x     
 
#     def forward(self, x):
#         x = self.forward_linear(x)
#         return torch.sigmoid(x)      
    
# class Combine_Stack_with_attn(nn.Module):
#     def __init__(self, model_size, output_features, w_size, layers=2, cat_feat=True):
#         super().__init__()
#         self.w_size = w_size
        
        
#         if layers==1:
#             self.attention = Attention(N_BINS, model_size, N_BINS)    
#             if cat_feat==True:
#                 self.linear1 = nn.Linear(N_BINS+N_BINS, output_features)
#                 self.forward_linear = self.forward_linear1_cat_feat
#             else:    
#                 self.linear1 = nn.Linear(N_BINS, output_features)
#                 self.forward_linear = self.forward_linear1_feat
#         elif layers==2:
#             self.attention = Attention(N_BINS, model_size, N_BINS)    
#             self.linear1 = nn.Linear(N_BINS, model_size)
#             if cat_feat==True:
#                 self.linear2 = nn.Linear(model_size+N_BINS, output_features)
#                 self.forward_linear = self.forward_linear2
#             else:
#                 raise Exception("cat_feat can't be used under layers=2") 
#         else:
#             print('Error: Layers can be only 1 or 2')

#     def forward_linear1_cat_feat(self, x, weighted):
#         x = self.linear1(torch.cat((x, weighted), dim=-1))
#         return x
    
#     def forward_linear1_feat(self, x, weighted):
#         x = self.linear1(weighted)
#         return x    
    
#     def forward_linear2(self, x, weighted):
#         x = torch.relu(self.linear1(x))
#         x = self.linear2(torch.cat((x, weighted), dim=-1))
#         return x
      
            
#     def forward(self, x, encoder_outputs):
#         batch_size = encoder_outputs.shape[0]
#         seq_len = encoder_outputs.shape[1]
#         feature_size = encoder_outputs.shape[2]        
#         # encoder_outputs.shape = (batch, seq_len, features)
        
#         #Padding the sequence so that the local attention size is fixed and can be packed into a tensor
#         encoder_outputs = nn.functional.pad(encoder_outputs,
#                                             (0,0,self.w_size,self.w_size)) 
#         padded_seq_len = encoder_outputs.shape[1]        
#         windowed_encoder_outputs = torch.as_strided(encoder_outputs, 
#                                                     (batch_size, seq_len, self.w_size*2+1, feature_size), 
#                                                     (padded_seq_len*feature_size, feature_size, feature_size, 1))

#         a = self.attention(x, windowed_encoder_outputs)
#         a = a.unsqueeze(2)
#         # a =                (batch, seq_len, 1, att_len)
#         # windowed_outputs = (batch, seq_len, att_len, features)
#         weighted = torch.matmul(a, windowed_encoder_outputs).squeeze(2) # remove the extra dim       
#         x = self.forward_linear(x, weighted)
        
#         return torch.sigmoid(x), a

    
# class SimpleModel(nn.Module):
#     def __init__(self, input_features, output_features, model_complexity=128, log=True, mode='imagewise', spec='Mel', w_size=30, attention=True, layers=2, cat_feat=True):
#         super().__init__()
#         self.layers = layers
#         self.w_size=w_size
#         self.log = log
#         self.normalize = Normalization(mode)        
#         self.spectrogram = Spectrogram.MelSpectrogram(sr=SAMPLE_RATE, win_length=WINDOW_LENGTH, n_mels=N_BINS, 
#                                                       hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
#                                                       trainable_mel=False, trainable_STFT=False)
        
#         model_size = model_complexity
#         if attention==True:
            
#             self.combined_stack = Combine_Stack_with_attn(model_size, output_features, w_size, layers, cat_feat)
#             self.forward = self.forward_w_attention
#             self.run_on_batch = self.run_on_batch_w_attention
#         elif attention==False:
#             self.combined_stack = Combine_Stack(model_size, output_features, layers)
#             self.forward = self.forward_wo_attention
#             self.run_on_batch = self.run_on_batch_wo_attention
        
#     def forward_w_attention(self, spec):
#         frame_pred, a = self.combined_stack(spec, spec) # Attenting on features
#         return frame_pred, a 
#     def forward_wo_attention(self, spec):
#         frame_pred = self.combined_stack(spec) # Attenting on features
#         return frame_pred

               
#     def run_on_batch_w_attention(self, batch):
#         audio_label = batch['audio']
#         onset_label = batch['onset']
# #         offset_label = batch['offset']
#         frame_label = batch['frame']
# #         velocity_label = batch['velocity']
  
#         spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
#         if self.log:
#             spec = torch.log(spec + 1e-5)
#         # print(f'spec shape = {spec.shape}')
#         spec = self.normalize.transform(spec)
#         spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
#         frame_pred, a = self(spec)
#         predictions = {
#             'onset': frame_pred.reshape(*frame_label.shape),
#             'frame': frame_pred.reshape(*frame_label.shape),
#             'attention': a
#         }
#         losses = {
#             'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
#         }           
            
#         return predictions, losses, spec
    
#     def run_on_batch_wo_attention(self, batch):
#         audio_label = batch['audio']
#         onset_label = batch['onset']
# #         offset_label = batch['offset']
#         frame_label = batch['frame']
# #         velocity_label = batch['velocity']
  
#         spec = self.spectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
#         if self.log:
#             spec = torch.log(spec + 1e-5)
#         # print(f'spec shape = {spec.shape}')
#         spec = self.normalize.transform(spec)
#         spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
#         frame_pred = self(spec)
#         predictions = {
#             'onset': frame_pred.reshape(*frame_label.shape),
#             'frame': frame_pred.reshape(*frame_label.shape),
#         }
#         losses = {
#             'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
#         }         
            
#         return predictions, losses, spec    
    
#     def feed_audio(self, audio):
# #         velocity_label = batch['velocity']
  
#         spec = self.spectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]) # x = torch.rand(8,229, 640)
#         if self.log:
#             spec = torch.log(spec + 1e-5)
#         # print(f'spec shape = {spec.shape}')
#         spec = self.normalize.transform(spec)
#         spec = spec.transpose(-1,-2) # swap spec bins with timesteps so that it fits LSTM later # shape (8,640,229)
        
        
#         onset_pred, activation_pred, frame_pred, a = self(spec)
        
#         predictions = {
#             'onset': onset_pred,
# #             'offset': offset_pred.reshape(*offset_label.shape),
#             'activation': activation_pred,
#             'frame': frame_pred,
#             'attention': a
# #             'velocity': velocity_pred.reshape(*velocity_label.shape)
#         }

#         return predictions, spec

#     def load_my_state_dict(self, state_dict):
 
#         own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name not in own_state:
#                  continue
#             if isinstance(param, nn.Parameter):
#                 # backwards compatibility for serialized parameters
#                 param = param.data
#             own_state[name].copy_(param)   
