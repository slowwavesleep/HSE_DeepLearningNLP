from src.data import basic_load
from src.tokenization import train_bpe, batch_tokenize
from src.datasets import QAData
from src.models import MyNet
from src.training import training_cycle
from src.generation import Generator
from torch.utils.data import DataLoader
import torch
import youtokentome as yttm


SOURCE_TRAIN_PATH = 'data/source.train'
SOURCE_DEV_PATH = 'data/source.dev'
SOURCE_TEST_PATH = 'data/source.test'
TARGET_TRAIN_PATH = 'data/target.train'
TARGET_DEV_PATH = 'data/target.dev'
TARGET_TEST_PATH = 'data/target.test'
BPE_TEXT_PATH = 'models/bpe_raw.txt'
BPE_MODEL_PATH = 'models/bpe_qa.model'
RESPONSES_GREEDY_PATH = 'results/greedy_responses.txt'
RESPONSES_NUCLEUS_PATH = 'results/nucleus_responses.txt'
VOCAB_SIZE = 7000
MAX_SOURCE_LEN = 40
MAX_TARGET_LEN = 40
PAD_INDEX = 0
UNK_INDEX = 1
UNK_INDEX = 2
EOS_INDEX = 3
BATCH_SIZE = 512
EMB_DIM = 512
HIDDEN_SIZE = 512


TRAIN_BPE = True
TRAIN_NET = True
GENERATE = True

source_train = basic_load(SOURCE_TRAIN_PATH)
target_train = basic_load(TARGET_TRAIN_PATH)
source_dev = basic_load(SOURCE_DEV_PATH)
target_dev = basic_load(TARGET_DEV_PATH)

assert len(source_train) == len(target_train)
assert len(source_dev) == len(target_dev)

if TRAIN_BPE:

    train_bpe(sentences=source_train,
              bpe_text_path=BPE_TEXT_PATH,
              bpe_model_path=BPE_MODEL_PATH,
              vocab_size=VOCAB_SIZE)

bpe = yttm.BPE(model=BPE_MODEL_PATH)

source_train_tokenized = batch_tokenize(source_train, bpe, bos=False, eos=False)
source_dev_tokenized = batch_tokenize(source_dev, bpe, bos=False, eos=False)
target_train_tokenized = batch_tokenize(target_train, bpe, bos=False, eos=False)
target_dev_tokenized = batch_tokenize(target_dev, bpe, bos=False, eos=False)

assert len(source_train_tokenized) == len(target_train_tokenized)
assert len(source_dev_tokenized) == len(target_dev_tokenized)

train_ds = QAData(source_train_tokenized,
                  target_train_tokenized,
                  MAX_SOURCE_LEN,
                  MAX_TARGET_LEN)

valid_ds = QAData(source_dev_tokenized,
                  target_dev_tokenized,
                  MAX_SOURCE_LEN,
                  MAX_TARGET_LEN)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, BATCH_SIZE)

GPU = torch.cuda.is_available()

if GPU:
    print('Using GPU...')
    device = torch.device('cuda')
else:
    raise NotImplementedError

model = MyNet(emb_dim=EMB_DIM,
              hidden_size=HIDDEN_SIZE,
              vocab_size=VOCAB_SIZE,
              dropout=0.4,
              weight_tying=True)

model.to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
optimizer = torch.optim.Adam(params=model.parameters())

if TRAIN_NET:

    training_cycle(model, train_loader, valid_loader, optimizer, criterion,
                   device,  3., 5)


if GENERATE:

    source_test = basic_load(SOURCE_TEST_PATH)[:100]
    target_test = basic_load(TARGET_TEST_PATH)[:100]

    model.load_state_dict(torch.load('models/best_language_model_state_dict.pth'))

    generator = Generator(bpe, model, device)
    generator.to_file(source_test, target_test, RESPONSES_GREEDY_PATH, 'greedy')
    generator.to_file(source_test, target_test, RESPONSES_NUCLEUS_PATH, 'nucleus')


