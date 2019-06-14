import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

class Attn(nn.Module):
    def __init__(self, h_dim, attn_num):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, attn_num),
            nn.ReLU(True),
            nn.Linear(attn_num,1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        tmp = encoder_outputs.contiguous().view(-1, self.h_dim)
        attn_ene = self.main(tmp) # (b, s, h) -> (b * s, 1)
        return F.softmax(attn_ene.view(b_size, -1), dim=1).unsqueeze(2) # (b*s, 1) -> (b, s, 1)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=True)

    def forward(self, x):
        # 璁剧疆鍒濆鐘舵€乭_0涓巆_0鐨勭姸鎬佹槸鍒濆鐨勭姸鎬侊紝涓€鑸缃负0锛屽昂瀵告槸,x.size(0)
        h0 = (5 * torch.ones(self.num_layers * 2, x.size(0), self.hidden_size)).cuda()
        c0 = (torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).cuda()

        # Forward propagate RNN
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  # 閫佸叆涓€涓垵濮嬬殑x鍊硷紝浣滀负杈撳叆浠ュ強(h0, c0)

        # Decode hidden state of last time step
        # out = self.fc(out[:, -1, :])  # output涔熸槸batch_first, 瀹為檯涓奾_n涓巆_n骞朵笉鏄痓atch_first
        return out

class AttnClassifier(nn.Module):
    def __init__(self, h_dim, c_num, attn_num):
        super(AttnClassifier, self).__init__()
        self.attn = Attn(h_dim, attn_num)
        self.main = nn.Linear(h_dim, c_num)

    def forward(self, encoder_outputs):
        attns = self.attn(encoder_outputs)  # (b, s, 1)
        feats = (encoder_outputs * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        # return F.log_softmax(self.main(feats)), attns
        return self.main(feats), attns


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global count
count = 0
global rnn, classifier

def predict(data, true_ids, class_num):
    global count
    global rnn
    global classifier
    if count == 0:
        rnn, classifier = load_model(class_num)
        count += 1
    test_data = data_process(data, 20)
    rnn.eval()
    with torch.no_grad():
        test_data = torch.FloatTensor(test_data).to(device)
        encoder_outputs = rnn(test_data)
        out, attn = classifier(encoder_outputs)
        out_softmax = F.softmax(out, dim=1)
        out_softmax_sum = out_softmax.sum(0)
        t = [round(out_softmax_sum.cpu().numpy()[i], 4) for i in range(class_num)]

        id = t.index(max(t))
        return true_ids[id]

def load_model(num_class):
    couple_length = 20
    input_size = 3  # 杈撳叆鏁版嵁鐨勭壒寰佺淮搴?
    hidden_size = 400  # 闅愯棌灞傜殑size
    if num_class == 10:
        num_layers = 2  # 鏈夊灏戝眰
        attn_num = 64
        encoder_dir = './encoder_10_attention_400.pkl'
        rnn = RNN(input_size, hidden_size, num_layers, num_class).cuda()
        rnn.load_state_dict(torch.load(encoder_dir))
        atten_dir = './classifier_10_attention_400.pkl'
        classifier = AttnClassifier(hidden_size * 2, num_class, attn_num).cuda()
        classifier.load_state_dict(torch.load(atten_dir))
    elif num_class == 107:
        num_layers = 3  # 鏈夊灏戝眰
        attn_num = 128
        encoder_dir = './encoder_107_attention.pkl'
        rnn = RNN(input_size, hidden_size, num_layers, num_class).cuda()
        rnn.load_state_dict(torch.load(encoder_dir))
        atten_dir = './classifier_107_attention.pkl'
        classifier = AttnClassifier(hidden_size * 2, num_class, attn_num).cuda()
        classifier.load_state_dict(torch.load(atten_dir))

    else:
        print('Please choose 10/107 classify Task')
        sys.exit(0)
    return rnn, classifier



def data_process(data, couple_length):
    def flag(i, x, p):
        t = data[i][x][p][:]
        t.append(0) if p == 0 else t.append(1)
        return t

    S = []
    [S.append(flag(i, x, p)) for i in range(len(data)) for x in range(len(data[i])) for p in range(len(data[i][x])) if len(data[i][x]) > 1 or len(data[i]) > 1]
    delta_S = [[S[i+1][0]-S[i][0], S[i+1][1]-S[i][1], S[i+1][2]] for i in range(len(S) - 1)]


    delta_S_len = len(delta_S)

    seq_length = [100] * couple_length
    test_data = []
    index_all = []
    for i in range(couple_length):
        index = np.random.randint(0, delta_S_len - seq_length[i])
        index_all.append(index)
        test_data.append(delta_S[index: index + seq_length[i]])
    return np.array(test_data)
