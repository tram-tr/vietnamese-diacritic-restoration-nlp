import torch
import pickle
import utils
import argparse
import dataloader
import torch.utils
import os

# torch.set_default_device('cuda')

train_tone = 'data/train.tone'
train_notone = 'data/train.notone'
dev_tone = 'data/dev.tone'
dev_notone = 'data/dev.notone'
test_tone = 'data/test.tone'
test_notone = 'data/test.notone'

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

class LSTM(torch.nn.Module):
    def __init__(self, src_size, trg_size, dims, bidirectional=False):
        super().__init__()
        self.embed = torch.nn.Embedding(src_size, dims)
        self.ndirection = 2 if bidirectional else 1
        self.lstm = torch.nn.LSTM(dims, dims, bidirectional=bidirectional, batch_first=True)
        self.out = torch.nn.Linear(dims * self.ndirection, trg_size)
    
    def forward(self, src):
        embedded = self.embed(src)
        o, _ = self.lstm(embedded)
        o = self.out(o)
        return o

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', action='store_true', default=False)
    parser.add_argument('-bidirectional', action='store_true', default=False)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-num_epochs', type=int, default=100)
    args = parser.parse_args()

    print('loading tokenizer...')
    tokenizer = torch.load('src/tokenizer_standarized.h5')
    src_vocab = tokenizer['notone']
    trg_vocab = tokenizer['tone']

    print('loading train dataset...')
    train_dataset = utils.create_dataset(train_notone, train_tone)
    train_iter = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=32, shuffle=True)

    print('loading dev dataset...')
    dev_dataset = utils.create_dataset(dev_notone, dev_tone)
    dev_iter = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=32, shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    model = LSTM(len(src_vocab.word_index) + 1, len(trg_vocab.word_index) + 1, 256, bidirectional=args.bidirectional)
    if device.type == 'cuda':
        model = model.cuda()
    o = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)

    outdir = '../out/models'
    print(f'training lstm for {args.num_epochs} epochs....') if not args.bidirectional else print(f'training bi-lstm for {args.num_epochs} epochs...')
    if not args.bidirectional:
        model_name = 'lstm-model.h5'
    else:
        model_name = 'bilstm-model.h5'

    best_dev_loss = float('inf')
    best_model_path = os.path.join(outdir, f'{model_name}')

    for epoch in range(1, args.num_epochs + 1):
        print(f"epoch {epoch:02d}/{args.num_epochs}")
        print(f"\nepoch {epoch:02d}/{args.num_epochs}", flush=True)

        if not args.bidirectional:
            save_path = os.path.join(outdir + '/lstm', f'epoch_{epoch:02d}.h5')
        else:
            save_path = os.path.join(outdir + '/bilstm', f'epoch_{epoch:02d}.h5')

        # train the model and calculate training loss
        total_loss = 0.0
        total_item = 0

        model.train()

        with tqdm(total=len(train_iter)) as pbar:
            for src, trg in train_iter: 
                if device.type=='cuda':
                    src = src.cuda()
                    trg = trg.cuda()

                o.zero_grad()
                preds = model(src)
                y = trg.contiguous().view(-1)
                loss = torch.nn.functional.cross_entropy(preds.view(-1, preds.size(-1)), y, ignore_index=0)
                loss.backward()
                o.step()
                
                total_loss += loss.item()
                total_item += trg.size(0)

                pbar.update(1)
                pbar.set_description("loss = %.8f" % (total_loss/total_item))
                
        state = {
            'model': model.state_dict(),
            'optimizer': o.state_dict()
        }

        torch.save(state, save_path)
        train_loss = total_loss / total_item
        print(f"train loss: {train_loss:.8f}")

        # evaluate the model and calculate validation loss
        total_loss = 0.0
        total_item = 0
        model.eval()
        with torch.no_grad(), tqdm(total=len(dev_iter)) as pbar:
            for src, trg in dev_iter:
                if device.type=='cuda':
                    src = src.cuda()
                    trg = trg.cuda()

                preds = model(src)
                y = trg.contiguous().view(-1)
                loss = torch.nn.functional.cross_entropy(preds.view(-1, preds.size(-1)), y, ignore_index=0)
                
                total_loss += loss.item()
                total_item += src.size(0)

                pbar.update(1)
                pbar.set_description("val_loss = %.8f" % (total_loss/total_item))
        
        dev_loss = total_loss / total_item
        print(f"validation loss: {dev_loss:.8f}\n")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            model.save(best_model_path)
            print(f"saving best model with validation loss: {best_dev_loss:.8f}")



    