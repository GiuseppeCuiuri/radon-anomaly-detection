"""
Author: Naji NAJARI
Software description: Robust Autoencoders for Unsupervised Anomaly Detection
"""

import torch
import torch.nn as nn
# from torchvision import  transforms
import torch.nn.functional as F
# import torch.distributions.transforms as transform
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd 
import time
import sys



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class RADON(nn.Module):

    def __init__(self,  input_dim, dim1, dim2):
        super(RADON, self).__init__()

        self.input_dim = input_dim
        self.dim1 = dim1
        self.dim2 = dim2
        
        #encoder
        self.fc1 = nn.Linear(input_dim, dim1)
        self.fc2 = nn.Linear(dim1, dim2)
	#decoder
        self.fc3 = nn.Linear(dim2, dim1)
        self.fc4 = nn.Linear(dim1, input_dim)

        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x)) 
        return x        


def fit(model, X, bin_estimator = "knuth",  epochs=20, projection_after_epochs = 50, batch_size=256, lr = 0.001, verbose=False):

	criterion = nn.CosineSimilarity(dim=1, eps=1e-5)
	model.to(device)
	losses = []
	val_losses = []
	th = []
	th_L = []
	th_S = []
	epsilon = 0.01
	cvg = []
	list_bins_L = []
	list_bins_S = []
	X = torch.FloatTensor(X.values).to(device)
	S =  torch.FloatTensor(np.zeros(X.shape)).to(device)
	L = X
	train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(L, S),batch_size=batch_size)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)


	for epoch in range(1, epochs+1):
		train_loss = 0.0
		validation_loss = 0.0
		model.train()

		for l,s in train_loader:
			optimizer.zero_grad()
			loss = ((1-criterion(model.forward(l), l))**2 +  (criterion(model.forward(s), s))**2).mean() 
			loss.backward()
			optimizer.step()
			train_loss += loss.item()*l.size(0)
				                       
        # Project the data to update the subsets L, S, and U                               
		if epoch %projection_after_epochs == 0 : 
			with torch.no_grad():
				model.eval()
				RE_X = criterion(model.forward(X), X).detach().cpu().numpy()
				if bin_estimator == 'knuth':
					bins =knuth_bin_width(RE_X,10,250,1)
				else :
					bins=np.histogram_bin_edges(RE_X, bins=bin_estimator)
				plt.figure(figsize=(10,7))
				histogram_X = plt.hist(RE_X,bins=bins)
				hist_X = histogram_X[0][::-1].astype(int)

				if np.linalg.norm(S.cpu().numpy(),ord=2)==0 :   

					bin_centers_X = ((histogram_X[1][1:] + histogram_X[1][:-1])/2)[::-1]
					bins = len(hist_X)
					list_bins_L.append(bins)
					idx,threshold = unimodal(hist_X,bin_centers_X)
					threshold_L = threshold 
					plt.axvline(threshold_L,color='red')
					idx_L = bins-idx
					idx_S = idx
					threshold_S = threshold     		    
		
				else:
					bins = len(hist_X)
					list_bins_L.append(bins)
					hist_L = histogram_X[0][::-1][:bins//2].astype(int)
					bin_centers_L = ((histogram_X[1][1:] + histogram_X[1][:-1])/2)[::-1][:bins//2]
					hist_S = histogram_X[0][:bins//2].astype(int)
					bin_centers_S = ((histogram_X[1][1:] + histogram_X[1][:-1])/2)[:bins//2]
					idx_L,threshold_L = unimodal(hist_L,bin_centers_L)
					idx_S,threshold_S = unimodal(hist_S,bin_centers_S)
					plt.axvline(threshold_L,color='red')
					plt.axvline(threshold_S,color='red')
					threshold = 0.5 * (histogram_X[1][-1] -  histogram_X[1][0])
					
				Y_L = np.where(criterion(model.forward(X), X).cpu().numpy() > threshold_L)[0]
				Y_S = np.where(criterion(model.forward(X), X).cpu().numpy() < threshold_S)[0]

				L = torch.from_numpy(np.array([X[i].cpu().numpy() for i in Y_L])).to(device) 
				S = torch.from_numpy(np.array([X[i].cpu().numpy() for i in Y_S])).to(device) 
				if (S.shape[0]==0):
					S =  torch.FloatTensor(np.zeros(L.shape)).to(device)
				else : 
					S = torch.cat((S,torch.FloatTensor(np.zeros([L.shape[0]-S.shape[0],L.shape[1]])).to(device) ))
				print('threshold_L {}'.format(threshold_L))
				print('threshold_S {}'.format(threshold_S))
				th.append(threshold)
				th_L.append(threshold_L)
				th_S.append(threshold_S)      
				plt.show()


		train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(L, S),batch_size=batch_size)
		losses.append(train_loss)
		print (" Epoch : ", epoch ,", Train cost : ", train_loss  )


	return  losses , th, th_L, th_S



# The unimodal thresholding strategy 
def unimodal(hist,bin_centers):
    pos_peak = np.argmax(hist)
    peak_max = hist[pos_peak]
    p1 = np.array([pos_peak, peak_max])

    if len( np.where(hist != 0)[0]>0):
        last_zeroBin = np.where(hist != 0)[0][-1]+1
    else : 
        last_zeroBin = len(hist)-1
    p2 = np.array([last_zeroBin, 0])
    best_idx = -1;
    max_dist = -1;
    for x0 in range(pos_peak,last_zeroBin):
        y0 = hist[x0];
        a = p1 - p2;
        b = [x0,y0] - p2;
        cross_ab = a[0]*b[1]-b[0]*a[1]
        d = np.linalg.norm(cross_ab)/np.linalg.norm(a);
        if(d>max_dist):
            best_idx = x0;
            max_dist = d;
    return best_idx,bin_centers[best_idx]
    
    
    

#Knuth bin estimator
def knuth_bin_width(data, N_MIN, N_MAX, step, return_bins=False, quiet=True):
    from scipy import optimize
    knuthF = _KnuthF(data)
    cost=[]
    B = range(N_MIN,N_MAX+1,step)
    for i in range(N_MIN,N_MAX+1,step):
        cost.append(knuthF.eval(i))

    bins=B[np.argmin(cost)]
    return bins


class _KnuthF:
    def __init__(self, data):
        self.data = np.array(data, copy=True)
        if self.data.ndim != 1:
            raise ValueError("data should be 1-dimensional")
        self.data.sort()
        self.n = self.data.size
        from scipy import special
        self.gammaln = special.gammaln

    def bins(self, M):
        """Return the bin edges given a width dx"""
        return np.linspace(self.data[0], self.data[-1], int(M) + 1)

    def __call__(self, M):
        return self.eval(M)

    def eval(self, M):
        M = int(M)

        if M <= 0:
            return np.inf

        bins = self.bins(M)
        nk, bins = np.histogram(self.data, bins)

        return -(self.n * np.log(M) +
                 self.gammaln(0.5 * M) -
                 M * self.gammaln(0.5) -
                 self.gammaln(self.n + 0.5 * M) +
                 np.sum(self.gammaln(nk + 0.5)))
