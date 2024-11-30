import os
import string
import math
import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image
import editdistance
from tqdm import tqdm
from config import ALPHABET, CHANNELS, WIDTH, HEIGHT, DEVICE, BATCH_SIZE


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale = torch.nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

def process_data(image_dir, labels_dir, ignore=[]):
    """
    params
    ---
    image_dir : str
      path to directory with images

    labels_dir : str
      path to tsv file with labels

    returns
    ---

    img2label : dict
      keys are names of images and values are correspondent labels

    chars : list
      all unique chars used in data

    all_labels : list
    """

    chars = []
    img2label = dict()

    with open(labels_dir, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        try:
            filename, label = line.strip().split('\t')
            if not any(item in label for item in ignore):
                img2label[os.path.join(image_dir, filename)] = label
                chars.extend(char for char in label if char not in chars)
        except ValueError:
            print('Bad line:', line)

    all_labels = sorted(set(img2label.values()))
    chars.sort()
    chars = ['PAD', 'SOS'] + chars + ['EOS']

    return img2label, chars, all_labels

def indices_to_text(indexes, idx2char):
    text = "".join([idx2char[i] for i in indexes])
    text = text.replace('EOS', '').replace('PAD', '').replace('SOS', '')
    return text

def char_error_rate(p_seq1, p_seq2):
    """
    params
    ---
    p_seq1 : str
    p_seq2 : str

    returns
    ---
    cer : float
    """
    p_vocab = set(p_seq1 + p_seq2)
    p2c = {p: chr(i) for i, p in enumerate(p_vocab)}
    c_seq1 = ''.join(p2c[p] for p in p_seq1)
    c_seq2 = ''.join(p2c[p] for p in p_seq2)

    if not c_seq1 or not c_seq2:
        return 1.0 if c_seq1 or c_seq2 else 0.0

    return editdistance.eval(c_seq1, c_seq2) / max(len(c_seq1), len(c_seq2))

def process_image(img):
    """
    params:
    ---
    img : np.array

    returns
    ---
    img : np.array
    """
    h, w, _ = img.shape
    new_h = HEIGHT
    new_w = int(w * (new_h / h))
    img = cv2.resize(img, (new_w, new_h))

    img = img.astype('float32')
    pad_w = WIDTH - new_w
    if pad_w > 0:
        img = np.pad(img, ((0, 0), (0, pad_w), (0, 0)), mode='constant', constant_values=255)
    else:
        img = cv2.resize(img, (WIDTH, new_h))

    return img

def generate_data(img_paths):
    """
    params
    ---
    names : list of str
        paths to images

    returns
    ---
    data_images : list of np.array
        images in np.array format
    """
    data_images = []
    for path in tqdm(img_paths):
        try:
            img = np.asarray(Image.open(path).convert('RGB'))
            img = process_image(img)
            data_images.append(img.astype('uint8'))
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return data_images


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, criterion, loader, case=True, punct=True):
    """
    params
    ---
    model : nn.Module
    criterion : nn.Object
    loader : torch.utils.data.DataLoader

    returns
    ---
    epoch_loss / len(loader) : float
        overall loss
    """
    model.eval()
    metrics = {'loss': 0, 'wer': 0, 'cer': 0}
    result = {'true': [], 'predicted': [], 'cer': []}
    with torch.no_grad():
        for (src, trg) in loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            logits = model(src, trg[:-1, :])
            loss = criterion(logits.view(-1, logits.shape[-1]), trg[1:, :].reshape(-1))
            out_indexes = model.predict(src)

            true_phrases = [indices_to_text(trg.T[i][1:], ALPHABET) for i in range(BATCH_SIZE)]
            pred_phrases = [indices_to_text(out_indexes[i], ALPHABET) for i in range(BATCH_SIZE)]

            if not case:
                true_phrases = [phrase.lower() for phrase in true_phrases]
                pred_phrases = [phrase.lower() for phrase in pred_phrases]
            if not punct:
                true_phrases = [phrase.translate(str.maketrans('', '', string.punctuation)) \
                                for phrase in true_phrases]
                pred_phrases = [phrase.translate(str.maketrans('', '', string.punctuation)) \
                                for phrase in pred_phrases]

            metrics['loss'] += loss.item()
            metrics['cer'] += sum([char_error_rate(true_phrases[i], pred_phrases[i]) \
                                   for i in range(BATCH_SIZE)]) / BATCH_SIZE
            metrics['wer'] += sum([int(true_phrases[i] != pred_phrases[i]) \
                                   for i in range(BATCH_SIZE)]) / BATCH_SIZE

            for i in range(len(true_phrases)):
                result['true'].append(true_phrases[i])
                result['predicted'].append(pred_phrases[i])
                result['cer'].append(char_error_rate(true_phrases[i], pred_phrases[i]))

    for key in metrics:
        metrics[key] /= len(loader)

    return metrics, result

def prediction(model, test_dir, char2idx, idx2char):
    """
    params
    ---
    model : nn.Module
    test_dir : str
        path to directory with images
    char2idx : dict
        map from chars to indicies
    idx2char : dict
        map from indicies to chars

    returns
    ---
    preds : dict
        key : name of image in directory
        value : dict with keys ['p_value', 'predicted_label']
    """
    preds = {}
    os.makedirs('/output', exist_ok=True)
    model.eval()

    with torch.no_grad():
        for filename in os.listdir(test_dir):
            img = Image.open(os.path.join(test_dir, filename)).convert('RGB')

            img = process_image(np.asarray(img)).astype('uint8')
            img = img / img.max()
            img = np.transpose(img, (2, 0, 1))

            src = torch.FloatTensor(img).unsqueeze(0).to(DEVICE)
            if CHANNELS == 1:
                src = transforms.Grayscale(CHANNELS)(src)
            out_indexes = model.predict(src)
            pred = indices_to_text(out_indexes[0], idx2char)
            preds[filename] = pred

    return preds


class ToTensor(object):
    def __init__(self, X_type=None, Y_type=None):
        self.X_type = X_type

    def __call__(self, X):
        X = X.transpose((2, 0, 1))
        X = torch.from_numpy(X)
        if self.X_type is not None:
            X = X.type(self.X_type)
        return X


def log_config(model):
    print('transformer layers:', model.enc_layers)
    print('transformer heads:', model.transformer.nhead)
    print('hidden dim:', model.decoder.embedding_dim)
    print('num classes:', model.decoder.num_embeddings)
    print('backbone:', model.backbone_name)
    print('dropout:', model.pos_encoder.dropout.p)

def log_config(model):
    print('transformer layers:', model.enc_layers)
    print('transformer heads:', model.transformer.nhead)
    print('hidden dim:', model.decoder.embedding_dim)
    print('num classes:', model.decoder.num_embeddings)
    print('backbone:', model.backbone_name)
    print('dropout:', model.pos_encoder.dropout.p)
    print(f'{count_parameters(model):,} trainable parameters')


def log_metrics(metrics, path_to_logs=None):
    if path_to_logs is not None:
        with open(path_to_logs, 'a') as f:
            if metrics['epoch'] == 1:
                f.write('Epoch\tTrain_loss\tValid_loss\tCER\tWER\tTime\n')
            f.write(f"{metrics['epoch']}\t{metrics['train_loss']}\t{metrics['loss']}\t{metrics['cer']}\t{metrics['wer']}\t{metrics['time']}\n")

    if metrics['epoch'] == 1:
        print('Epoch   Train_loss   Valid_loss   CER   WER    Time    LR')
        print('-----   -----------  ----------   ---   ---    ----    ---')
    print('{:02d}       {:.2f}         {:.2f}       {:.2f}   {:.2f}   {:.2f}   {:e}'.format(
        metrics['epoch'], metrics['train_loss'], metrics['loss'], metrics['cer'],
        metrics['wer'], metrics['time'], metrics['lr']
    ))

    if path_to_logs != None:
      f.write(str(metrics['epoch'])+'\t'+str(metrics['train_loss'])+'\t'+str(metrics['loss'])+'\t'+str(metrics['cer'])+'\t'+str(metrics['wer'])+'\t'+str(metrics['time'])+'\n')
      f.close()
