import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
from sklearn.neighbors import NearestNeighbors as KNN



class KNearestNeighbor(Function):
  @staticmethod
  def forward(self, ref, query):
    ref = ref.float().squeeze(0).transpose(0,1).cpu()
    query = query.float().squeeze(0).transpose(0, 1).cpu()
    knn = KNN(n_neighbors=1)

    knn.fit(ref)
    _,inds = knn.kneighbors(X=query,n_neighbors=1)

    inds = inds+1
    inds = torch.tensor(inds).transpose(0, 1).unsqueeze(0).cuda()

    return inds







if __name__ == '__main__':
  unittest.main()

