import torch
import torch.nn as nn
from nn_blocks import *
from torch import optim
import time


class DApredictModel(nn.Module):
    def __init__(self, device):
        super(DApredictModel, self).__init__()
        self.device = device

    def forward(self, X_da, Y_da, X_utt, Y_utt, step_size, turn,
                da_encoder, da_decoder, da_context, da_context_hidden,
                utt_encoder, utt_decoder, utt_context, utt_context_hidden,
                criterion, last, config):
        loss = 0
        if config['use_da']:
            da_encoder_hidden = da_encoder(X_da) # (batch_size, 1, DA_HIDDEN)
            da_context_output, da_context_hidden = da_context(da_encoder_hidden, da_context_hidden) # (batch_size, 1, DA_HIDDEN)
        if config['turn']:
            turn = turn.float()
            turn = turn.unsqueeze(1)  # (batch_size, 1, 1)


        if config['use_utt']:
            utt_encoder_hidden = utt_encoder.initHidden(step_size, self.device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden) # (batch_size, 1, UTT_HIDDEN)
            if config['turn']:
                if config['use_da']:
                    dec_hidden = torch.cat((da_context_output, utt_encoder_output, turn), dim=2) # (batch_size, 1, DEC_HIDDEN)
                else:
                    dec_hidden = torch.cat((utt_encoder_output, turn), dim=2)
            else:
                if config['use_da']:
                    dec_hidden = torch.cat((da_context_output, utt_encoder_output), dim=2)
                else:
                    dec_hidden = utt_encoder_output
        elif config['use_uttcontext']:
            utt_encoder_hidden = utt_encoder.initHidden(step_size, self.device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)
            if config['turn']:
                utt_encoder_hidden = torch.cat((utt_encoder_output, turn), dim=2)
            utt_context_output, utt_context_hidden = utt_context(utt_encoder_hidden, utt_context_hidden) # (batch_size, 1, UTT_HIDDEN)
            if config['use_da']:
                dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2) # (batch_size, 1, DEC_HIDDEN)
                if not config['use_dacontext']:
                    dec_hiden = torch.cat((da_encoder_hidden, utt_context_output), dim=2)
            else:
                dec_hidden = utt_context_output
        else:
            if config['turn']:
                dec_hidden = torch.cat((da_context_output, turn), dim=2)
            else:
                dec_hidden = da_context_output


        decoder_output = da_decoder(dec_hidden) # (batch_size, 1, DA_VOCAB)
        decoder_output = decoder_output.squeeze(1) # (batch_size, DA_VOCAB)
        Y_da = Y_da.squeeze()


        loss += criterion(decoder_output, Y_da)

        if last:
            loss.backward()
            return loss.item(), da_context_hidden, utt_context_hidden
        else:
            return da_context_hidden, utt_context_hidden

    def evaluate(self, X_da, Y_da, X_utt, Y_utt, turn,
                 da_encoder, da_decoder, da_context, da_context_hidden,
                 utt_encoder, utt_decoder, utt_context, utt_context_hidden,
                 criterion, config):
        with torch.no_grad():
            loss = 0
            if config['use_da']:
                da_encoder_hidden = da_encoder(X_da)
                da_context_output, da_context_hidden = da_context(da_encoder_hidden, da_context_hidden)

            if config['turn']:
                turn = turn.float()
                turn = turn.unsqueeze(1)  # (batch_size, 1, 1)

            if config['use_utt']:
                utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
                utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden) # (1, 1, UTT_HIDDEN)
                if config['turn']:
                    if config['use_da']:
                        dec_hidden = torch.cat((da_context_output, utt_encoder_output, turn), dim=2) # (1, 1, DEC_HIDDEN)
                    else:
                        dec_hidden = torch.cat((utt_encoder_output, turn), dim=2)
                else:
                    if config['use_da']:
                        dec_hidden = torch.cat((da_context_output, utt_encoder_output), dim=2)
                    else:
                        dec_hidden = utt_encoder_output
            elif config['use_uttcontext']:
                utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
                utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden)  # (1, 1, UTT_HIDDEN)
                if config['turn']:
                    utt_encoder_hidden = torch.cat((utt_encoder_output, turn), dim=2)
                utt_context_output, utt_context_hidden = utt_context(utt_encoder_hidden, utt_context_hidden) # (1, 1, UTT_HIDDEN)
                if config['use_da']:
                    dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2) # (1, 1, DEC_HIDDEN)
                    if not config['use_dacontext']:
                        dec_hidden = torch.cat((da_encoder_hidden, utt_context_output), dim=2)
                else:
                    dec_hidden = utt_context_output
            else:
                if config['turn']:
                    dec_hidden = torch.cat((da_context_output, turn), dim=2)
                else:
                    dec_hidden = da_context_output

            decoder_output = da_decoder(dec_hidden)
            decoder_output = decoder_output.squeeze(1)
            Y_da = Y_da.squeeze(0)

            loss += criterion(decoder_output, Y_da)

        return loss.item(), da_context_hidden, utt_context_hidden

    def predict(self, X_da, X_utt, turn, da_encoder, da_decoder, da_context, da_context_hidden,
                utt_encoder, utt_context, utt_context_hidden, config):
        with torch.no_grad():
            if config['use_da']:
                encoder_hidden=da_encoder(X_da)
                da_context_output, da_context_hidden = da_context(encoder_hidden, da_context_hidden)
        
            if config['turn']:
                turn = turn.float()
                turn = turn.unsqueeze(1)  # (batch_size, 1, 1)

            if config['use_utt']:
                utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
                seq_len = X_utt.size()[1]
                for ei in range(seq_len):
                    utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt[ei], utt_encoder_hidden) # (1, 1, UTT_HIDDEN)
                if config['turn']:
                    if config['use_da']:
                        dec_hidden = torch.cat((da_context_output, utt_encoder_output, turn), dim=2) # (1, 1, DEC_HIDDEN)
                    else:
                        dec_hidden = torch.cat((utt_encoder_output, turn), dim=2)
                else:
                    if config['use_da']:
                        dec_hidden = torch.cat((da_context_output, utt_encoder_output), dim=2)
                    else:
                        dec_hidden = utt_encoder_output

            elif config['use_uttcontext']:
                utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
                seq_len = X_utt.size()[1]
                for ei in range(seq_len):
                    utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt[ei], utt_encoder_hidden)  # (1, 1, UTT_HIDDEN)
                if config['turn']:
                    utt_encoder_hidden = torch.cat((utt_encoder_output, turn), dim=2)
                utt_context_output, utt_context_hidden = utt_context(utt_encoder_hidden, utt_context_hidden) # (1, 1, UTT_HIDDEN)
                if config['use_da']:
                    dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2) # (1, 1, DEC_HIDDEN)
                    if not config['use_dacontext']:
                        dec_hidden = torch.cat((encoder_hidden, utt_context_output), dim=2)
                else:
                    dec_hidden = utt_context_output
            else:
                if config['turn']:
                    dec_hidden = torch.cat((da_context_output, turn), dim=2)
                else:
                    dec_hidden = da_context_output

            decoder_output = da_decoder(dec_hidden)

        return decoder_output, da_context_hidden, utt_context_hidden


class EncoderDecoderModel(nn.Module):
    def __init__(self, device):
        super(EncoderDecoderModel, self).__init__()
        self.device = device

    def forward(self, X_da, Y_da, X_utt, Y_utt, step_size,
                da_encoder, da_context, da_decoder, da_context_hidden, 
                utt_encoder, utt_context, utt_decoder, utt_context_hidden, 
                criterion, last, config):
        loss = 0

        # Input Dialogue Act
        if config['use_da']:
            da_encoder_hidden = da_encoder(X_da)
            da_context_output, da_context_hidden = da_context(da_encoder_hidden, da_context_hidden)

        # Encode Utterance
        utt_encoder_hidden = utt_encoder.initHidden(step_size, self.device)
        utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)

        # Update Context Encoder
        utt_context_output, utt_context_hidden = utt_context(utt_encoder_output, utt_context_hidden) # (batch_size, 1, UTT_HIDDEN)

        # concat DA hidden and Utterance hidden
        if config['use_da']:
            da_dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2)
            utt_dec_hidden = torch.cat((da_context_hidden, utt_context_hidden), dim=2) # (batch_size, 1, DEC_HIDDEN)
        else:
            da_dec_hidden = utt_context_output
            utt_dec_hidden = utt_context_hidden

        # da_decoder_output = da_decoder(da_dec_hidden)
        # da_decoder_output = da_decoder_output.squeeze(1)

        # Decode
        utt_decoder_hidden = utt_dec_hidden
        for j in range(len(Y_utt[0]) - 1):
            prev_words = Y_utt[:, j].unsqueeze(1)
            preds, utt_decoder_hidden = utt_decoder(prev_words, utt_decoder_hidden)
            _, topi = preds.topk(1)
            loss += criterion(preds.view(-1, config['UTT_MAX_VOCAB']), Y_utt[:, j + 1])

        Y_da = Y_da.squeeze(0)

        # da_loss = criterion(da_decoder_output, Y_da)

        if last:
            loss.backward()
            return loss.item(), da_context_hidden, utt_context_hidden
        else:
            return loss, da_context_hidden, utt_context_hidden


    def evaluate(self, X_da, Y_da, X_utt, Y_utt,
                 da_encoder, da_decoder, da_context, da_context_hidden,
                 utt_encoder, utt_decoder, utt_context, utt_context_hidden,
                 criterion, config):
        with torch.no_grad():
            loss = 0
            if config['use_da']:
                da_encoder_hidden = da_encoder(X_da)
                da_context_output, da_context_hidden = da_context(da_encoder_hidden, da_context_hidden)

            # Encode Utterance
            utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt,
                                                                 utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)

            # Update Context Encoder
            utt_context_output, utt_context_hidden = utt_context(utt_encoder_output,
                                                                 utt_context_hidden)  # (batch_size, 1, UTT_HIDDEN)

            # concat DA hidden and Utterance hidden
            if config['use_da']:
                da_dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2)
                utt_dec_hidden = torch.cat((da_context_hidden, utt_context_hidden),
                                           dim=2)  # (batch_size, 1, DEC_HIDDEN)
            else:
                da_dec_hidden = utt_context_output
                utt_dec_hidden = utt_context_hidden

            # Y_da = Y_da.squeeze(0)
            # da_loss = criterion(decoder_output, Y_da)

            utt_decoder_hidden = utt_dec_hidden
            for j in range(len(Y_utt[0]) - 1):
                prev_words = Y_utt[:, j].unsqueeze(1)
                preds, utt_decoder_hidden = utt_decoder(prev_words, utt_decoder_hidden)
                _, topi = preds.topk(1)
                loss += criterion(preds.view(-1, config['UTT_MAX_VOCAB']), Y_utt[:, j + 1])

            # loss = loss + da_loss

        return loss.item(), da_context_hidden, utt_context_hidden


    def predict(self, X_da, X_utt,
                da_encoder, da_decoder, da_context, da_context_hidden,
                utt_encoder, utt_decoder, utt_context, utt_context_hidden,
                config, EOS_token=1, BOS_token=2):
        with torch.no_grad():
            if config['use_da']:
                da_encoder_hidden = da_encoder(X_da)
                da_context_output, da_context_hidden = da_context(da_encoder_hidden, da_context_hidden)

            # Encode Utterance
            utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt,
                                                                 utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)

            # Update Context Encoder
            utt_context_output, utt_context_hidden = utt_context(utt_encoder_output,
                                                                 utt_context_hidden)  # (batch_size, 1, UTT_HIDDEN)

            # concat DA hidden and Utterance hidden
            if config['use_da']:
                da_dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2)
                utt_dec_hidden = torch.cat((da_context_hidden, utt_context_hidden),
                                           dim=2)  # (batch_size, 1, DEC_HIDDEN)
            else:
                da_dec_hidden = utt_context_output
                utt_dec_hidden = utt_context_hidden

            # decoder_output = da_decoder(da_dec_hidden)

            utt_decoder_hidden = utt_dec_hidden
            prev_words = torch.tensor([[BOS_token]]).to(self.device)

            if config['beam_size']:
                pred_seq, utt_decoder_hidden = self._beam_decode(prev_words, utt_decoder, utt_decoder_hidden, EOS_token, config)
            else:
                pred_seq, utt_decoder_hidden = self._greedy_decode(prev_words, utt_decoder, utt_decoder_hidden, EOS_token, config)

        return pred_seq, da_context_hidden, utt_context_hidden

    def _greedy_decode(self, prev_words, decoder, decoder_hidden, EOS_token, config):
        pred_seq = []
        for _ in range(config['max_len']):
            preds, decoder_hidden = decoder(prev_words, decoder_hidden)
            _, topi = preds.topk(1)
            pred_seq.append(topi.item())
            prev_words = torch.tensor([[topi]]).to(self.device)
            if topi == EOS_token:
                break
        return pred_seq, decoder_hidden

    def _beam_decode(self, prev_words, decoder, decoder_hidden, EOS_token, config):
        pred_seq = []

        return pred_seq, decoder_hidden

class seq2seq(nn.Module):
    def __init__(self, device):
        super(seq2seq, self).__init__()
        self.device = device

    def forward(self, X, Y, encoder, decoder, context, step_size, criterion, config):
        loss = 0

        encoder_hidden = encoder.initHidden(step_size, self.device)
        encoder_output, encoder_hidden = encoder(X, encoder_hidden)

        context_hidden = context.initHidden(step_size, self.device)
        context_output, context_hidden = context(encoder_output, context_hidden)

        decoder_hidden = context_hidden
        print(decoder_hidden)
        for j in range(len(Y[0]) - 1):
            prev_words = Y[:, j].unsqueeze(1)
            preds, decoder_hidden = decoder(prev_words, decoder_hidden)
            _, topi = preds.topk(1)
            loss += criterion(preds.view(-1, config['UTT_MAX_VOCAB']), Y[:, j + 1])

        loss.backward()

        return loss.item()

    def evaluate(self, X, Y, encoder, decoder, context, criterion, config):
        with torch.no_grad():
            loss = 0

            encoder_hidden = encoder.initHidden(1, self.device)
            encoder_output, _ = encoder(X, encoder_hidden)

            context_hidden = context.initHidden(1, self.device)
            context_output, context_hidden = context(encoder_output, context_hidden)

            decoder_hidden = context_hidden
            for j in range(len(Y[0]) - 1):
                prev_words = Y[:, j].unsqueeze(1)
                preds, decoder_hidden = decoder(prev_words, decoder_hidden)
                _, topi = preds.topk(1)
                loss += criterion(preds.view(-1, config['UTT_MAX_VOCAB']), Y[:, j + 1])

            return loss.item()
    def predict(self, X, encoder, decoder, context, config, EOS_token, BOS_token):
        with torch.no_grad():
            encoder_hidden = encoder.initHidden(1, self.device)
            encoder_output, _ = encoder(X, encoder_hidden)

            context_hidden = context.initHidden(1, self.device)
            context_output, context_hidden = context(encoder_output, context_hidden)
            
            decoder_hidden = context_hidden
            prev_words = torch.tensor([[BOS_token]]).to(self.device)
            pred_seq = []
            for _ in range(config['max_len']):
                preds, decoder_hidden = decoder(prev_words, decoder_hidden)
                _, topi = preds.topk(1)
                pred_seq.append(topi.item())
                prev_words = torch.tensor([[topi]]).to(self.device)
                if topi == EOS_token:
                    break
        return pred_seq



