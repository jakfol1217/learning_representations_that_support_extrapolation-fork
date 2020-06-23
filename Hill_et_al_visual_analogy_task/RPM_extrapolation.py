import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
import time

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

from util import log

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

class RPM_dataset(Dataset):
	def __init__(self, root_dir, regime='interpolation', training_style='normal', train=True):
		self.root_dir = root_dir + regime + '/'
		self.regime = regime
		self.training_style = training_style
		self.train = train
		if self.train:
			fname_list = root_dir + self.regime + '_train_' + self.training_style + '.txt'
		else:
			fname_list = root_dir + self.regime + '_test_' + self.training_style + '.txt'
		self.all_filenames = open(fname_list, 'r').read().splitlines() 
		self.len = len(self.all_filenames)
	def __len__(self):
		return self.len
	def __getitem__(self, idx):
		fname = self.all_filenames[idx]
		sample = np.load(fname, allow_pickle=True)
		# Images
		images = torch.from_numpy(sample['image'].reshape(9, 160, 160)) / 255.
		# Resize
		images = F.interpolate(images.unsqueeze(0), size=80).squeeze()
		# Target
		target = torch.from_numpy(sample['target'])
		# Relation
		rel_encoded = torch.from_numpy(sample['relation_structure_encoded'])
		return images, target, rel_encoded

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		# Define layers
		self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
		self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
		self.relu = nn.ReLU()
		# Initialize parameters
		for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
			for name, param in layer.named_parameters():
				if 'weight' in name:
					nn.init.kaiming_normal_(param, nonlinearity='relu')
				elif 'bias' in name:
					nn.init.zeros_(param)	
	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.relu(self.conv4(x))
		z = x.view(x.shape[0], -1)
		return z

class RNN(nn.Module):
	def __init__(self, args):
		super(RNN, self).__init__()
		# Define layers
		self.hidden_size = args.recurrent_net_size
		self.rnn = nn.RNN(5*5*32, self.hidden_size, batch_first=True)
		self.out = nn.Linear(args.recurrent_net_size, 1)
		# Initialize parameters
		for layer in [self.rnn, self.out]:
			for name, param in layer.named_parameters():
				if 'weight' in name:
					nn.init.xavier_normal_(param)
				elif 'bias' in name:
					nn.init.zeros_(param)
	def forward(self, input_seq, device):
		# Initialize hidden state
		hidden = torch.zeros(1, input_seq.shape[0], self.hidden_size).to(device)
		# Apply recurrent network
		out, hidden = self.rnn(input_seq, hidden)
		# Apply output layer to final output
		final_out = out[:,-1,:]
		score = self.out(final_out)
		return score

class LSTM(nn.Module):
	def __init__(self, args):
		super(LSTM, self).__init__()
		# Define layers
		self.hidden_size = args.recurrent_net_size
		self.lstm = nn.LSTM(5*5*32, self.hidden_size, batch_first=True)
		self.out = nn.Linear(args.recurrent_net_size, 1)
		# Initialize parameters
		for layer in [self.lstm, self.out]:
			for name, param in layer.named_parameters():
				if 'weight' in name:
					nn.init.xavier_normal_(param)
				elif 'bias' in name:
					nn.init.zeros_(param)
	def forward(self, input_seq, device):
		# Initialize hidden state
		hidden = torch.zeros(1, input_seq.shape[0], self.hidden_size).to(device)
		cell_state = torch.zeros(1, input_seq.shape[0], self.hidden_size).to(device)
		# Apply recurrent network
		out, (hidden, cell_state) = self.lstm(input_seq, (hidden, cell_state))
		# Apply output layer to final output
		final_out = out[:,-1,:]
		score = self.out(final_out)
		return score

class Context_norm(nn.Module):
	def __init__(self):
		super(Context_norm, self).__init__()
		self.scale = nn.Parameter(torch.Tensor(5*5*32))
		nn.init.ones_(self.scale)
		self.shift = nn.Parameter(torch.Tensor(5*5*32))
		nn.init.zeros_(self.shift)
		self.eps = 1e-8
	def forward(self, seq):
		mu = seq.mean(1)
		sigma = (seq.var(1) + self.eps).sqrt()
		seq = (seq - mu.unsqueeze(1)) / sigma.unsqueeze(1)
		seq = (seq * self.scale) + self.shift
		return seq

class Batch_norm(nn.Module):
	def __init__(self):
		super(Batch_norm, self).__init__()
		self.scale = nn.Parameter(torch.Tensor(5*5*32))
		nn.init.ones_(self.scale)
		self.shift = nn.Parameter(torch.Tensor(5*5*32))
		nn.init.zeros_(self.shift)
		self.eps = 1e-8
	def forward(self, seq):
		mu = seq.mean(0)
		sigma = (seq.var(0) + self.eps).sqrt()
		seq = (seq - mu.unsqueeze(0)) / sigma.unsqueeze(0)
		seq = (seq * self.scale) + self.shift
		return seq

class Source_targ_norm(nn.Module):
	def __init__(self):
		super(Source_targ_norm, self).__init__()
		self.scale = nn.Parameter(torch.Tensor(5*5*32))
		nn.init.ones_(self.scale)
		self.shift = nn.Parameter(torch.Tensor(5*5*32))
		nn.init.zeros_(self.shift)
		self.eps = 1e-8
	def forward(self, seq):
		# Normalize source sequence
		source_seq = seq[:, :3, :]
		source_mu = source_seq.mean(1)
		source_sigma = (source_seq.var(1) + self.eps).sqrt()
		source_seq = (source_seq - source_mu.unsqueeze(1)) / source_sigma.unsqueeze(1)
		# Normalize target sequence
		targ_seq = seq[:, 3:, :]
		targ_mu = targ_seq.mean(1)
		targ_sigma = (targ_seq.var(1) + self.eps).sqrt()
		targ_seq = (targ_seq - targ_mu.unsqueeze(1)) / targ_sigma.unsqueeze(1)
		# Combine, scale/shift
		seq = torch.cat([source_seq, targ_seq], 1)
		seq = (seq * self.scale) + self.shift
		return seq

class Analogy_scoring_model(nn.Module):
	def __init__(self, encoder, norm, rnn):
		super(Analogy_scoring_model, self).__init__()
		self.encoder = encoder
		self.norm = norm
		self.rnn = rnn
	def get_score(self, seq, device):
		all_z = []
		for t in range(seq.shape[1]):
			x = seq[:,t,:,:].unsqueeze(1)
			z = self.encoder(x)
			all_z.append(z)
		all_z = torch.stack(all_z, dim=1)
		if self.norm is not None:
			all_z = self.norm(all_z)
		score = self.rnn(all_z, device)
		return score
	def forward(self, images, device):
		all_scores = []
		for c in range(5,9):
			seq = torch.cat([images[:,0:5,:,:], images[:,c,:,:].unsqueeze(1)], 1)
			score = self.get_score(seq, device)
			all_scores.append(score)
		all_scores = torch.cat(all_scores, dim=1)
		return all_scores

def train(args, scoring_model, device, optimizer, epoch, train_loader):
	# Create file for saving training progress
	train_prog_dir = './train_prog/'
	check_path(train_prog_dir)
	model_dir = train_prog_dir + args.model_name + '/'
	check_path(model_dir)
	train_prog_fname = model_dir + 'epoch_' + str(epoch) + '.txt'
	train_prog_f = open(train_prog_fname, 'w')
	train_prog_f.write('batch loss acc\n')
	# Set to training mode
	scoring_model.train()
	# Iterate over batches
	for batch_idx, (images, target, rel) in enumerate(train_loader):
		# Batch start time
		start_time = time.time()
		# Load data
		images = images.to(device)
		target = target.to(device)
		# Zero out gradients for optimizer 
		optimizer.zero_grad()
		# Get analogy scores
		scores = scoring_model(images, device)
		# Loss
		loss_fn = torch.nn.CrossEntropyLoss()
		loss = loss_fn(scores, target)
		# Update model
		loss.backward()
		optimizer.step()
		# Accuracy
		acc = torch.eq(scores.argmax(1), target).float().mean().item() * 100.0
		# Batch duration
		end_time = time.time()
		batch_dur = end_time - start_time
		# Report progress
		if batch_idx % args.log_interval == 0:
			log.info('[Epoch: ' + str(epoch) + '] ' + \
					 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_loader)) + '] ' + \
					 '[Loss = ' + '{:.4f}'.format(loss.item()) + '] ' + \
					 '[Accuracy = ' + '{:.2f}'.format(acc) + '] ' + \
					 '[' + '{:.3f}'.format(batch_dur) + ' sec/batch]')
			# Save progress to file
			train_prog_f.write(str(batch_idx) + ' ' +\
							   '{:.4f}'.format(loss.item()) + ' ' + \
							   '{:.2f}'.format(acc) + '\n')
	train_prog_f.close()

def save_model(args, scoring_model, epoch):
	log.info('Saving model, epoch ' + str(epoch) + '...')
	# Create directory
	model_dir = './saved_params/'
	check_path(model_dir)
	model_dir = model_dir + args.model_name + '/'
	check_path(model_dir)
	epoch_dir = model_dir + str(epoch) + '/'
	check_path(epoch_dir)
	# Save classifier
	scoring_model_fname = epoch_dir + 'scoring_model.pt'
	torch.save(scoring_model.state_dict(), scoring_model_fname)

def main():

	# Training settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--recurrent_net_type', type=str, default='rnn')
	parser.add_argument('--recurrent_net_size', type=int, default=64)
	parser.add_argument('--norm_type', type=str, default=None)
	parser.add_argument('--train_batch_size', type=int, default=32)
	parser.add_argument('--epochs', type=int, default=3)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--regime', type=str, default='extrapolation')
	parser.add_argument('--training_style', type=str, default='lbc')
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--log_interval', type=int, default=10)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--run', type=str, default='1')
	args = parser.parse_args()

	# Set up cuda	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# Create training/test sets
	dset_dir = './datasets/'
	log.info('Creating data loader for training set...')
	train_set = RPM_dataset(dset_dir, regime=args.regime, training_style=args.training_style, train=True)
	train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)

	# Create unique name for this model
	args.model_name = args.regime + '_'
	args.model_name += args.training_style + '_'
	args.model_name += args.recurrent_net_type + str(args.recurrent_net_size) + '_'
	if args.norm_type is not None:
		args.model_name += args.norm_type + '_'
	args.model_name += 'run' + args.run

	# Build model
	log.info('Building model...')
	encoder = Encoder().to(device)
	if args.recurrent_net_type == 'rnn':
		rnn = RNN(args).to(device)
	elif args.recurrent_net_type == 'lstm':
		rnn = LSTM(args).to(device)
	if args.norm_type == 'context_norm':
		norm = Context_norm().to(device)
	elif args.norm_type == 'batch_norm':
		norm = Batch_norm().to(device)
	elif args.norm_type == 'source_targ_norm':
		norm = Source_targ_norm().to(device)
	else:
		norm = None
	scoring_model = Analogy_scoring_model(encoder, norm, rnn).to(device)

	# Create optimizer
	log.info('Setting up optimizer...')
	optimizer = optim.Adam(scoring_model.parameters(), lr=args.lr)

	# Train
	log.info('Training begins...')
	for epoch in range(1, args.epochs + 1):
		# Training loop
		train(args, scoring_model, device, optimizer, epoch, train_loader)
		# Save model
		save_model(args, scoring_model, epoch)

if __name__ == '__main__':
	main()