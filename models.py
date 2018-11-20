import torch
import torch.nn as nn
from nn_blocks import *
from torch import optim
import time


class DAonlyModel(nn.Module):
    def __init__(self):
        super(DAonlyModel, self).__init__()

    def forward(self, X_tensor, Y_tensor,
              encoder, decoder, context, context_hidden,
              criterion, last):
        loss = 0

        encoder_hidden = encoder(X_tensor)

        context_output, context_hidden = context(encoder_hidden, context_hidden)

        decoder_output = decoder(context_hidden)
        decoder_output = decoder_output.squeeze(1)

        loss += criterion(decoder_output, Y_tensor)

        if last:
            loss.backward()
            return loss.item(), context_hidden
        else:
            return context_hidden

    def evaluate(self, X_tensor, Y_tensor,
                 encoder, decoder, context, context_hidden,
                 criterion):
        loss = 0
        encoder_hidden = encoder(X_tensor)
        context_output, context_hidden = context(encoder_hidden, context_hidden)
        decoder_output = decoder(context_hidden)
        decoder_output = decoder_output.squeeze(1)
        loss += criterion(decoder_output, Y_tensor)
        return loss.item(), context_hidden

    def predict(self, X_tensor, encoder, decoder, context, context_hidden):
        encoder_hidden=encoder(X_tensor)
        context_output, context_hidden = context(encoder_hidden, context_hidden)
        decoder_output = decoder(context_hidden)

        return decoder_output, context_hidden

class DAuttModel(nn.Module):
    def __init__(self):
        super(DAuttModel, self).__init__()

    def forward(self, X_da, Y_da,
              da_encoder, da_decoder, da_context, da_context_hidden,
              criterion, last):
        loss = 0

        encoder_hidden = da_encoder(X_da)

        context_output, context_hidden = da_context(encoder_hidden, da_context_hidden)

        decoder_output = da_decoder(context_hidden)
        decoder_output = decoder_output.squeeze(1)

        loss += criterion(decoder_output, Y_da)

        if last:
            loss.backward()
            return loss.item(), context_hidden
        else:
            return context_hidden

    def evaluate(self, X_da, Y_da,
                 da_encoder, da_decoder, da_context, da_context_hidden,
                 criterion):
        loss = 0
        encoder_hidden = da_encoder(X_da)
        context_output, context_hidden = da_context(encoder_hidden, da_context_hidden)
        decoder_output = da_decoder(context_hidden)
        decoder_output = decoder_output.squeeze(1)
        loss += criterion(decoder_output, Y_da)
        return loss.item(), context_hidden

    def predict(self, X_tensor, encoder, decoder, context, context_hidden):
        encoder_hidden=encoder(X_tensor)
        context_output, context_hidden = context(encoder_hidden, context_hidden)
        decoder_output = decoder(context_hidden)

        return decoder_output, context_hidden