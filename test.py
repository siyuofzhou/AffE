import sys
import numpy as np
import torch
from utils.euclidean import givens_rotations
N1 = 10
N2 = 15
d = 30

a = np.random.random([N1, d])
b = np.random.random([N2, d])
t1 = np.random.random([N2, d])
t2 = np.random.random([N2, d])
r1 = np.random.random([N1, d])
r2 = np.random.random([N1, d])

all1 = a.reshape([N1, 1, d]) - b.reshape(([1, N2, d]))
all2 = (t1.reshape([1, N2, d])*r1.reshape([N1, 1, d])-t2.reshape([1, N2, d])*r2.reshape([N1, 1, d]))
print(np.sum((all1+all2)**2,axis=-1)[0,0])


def euc_sqdistance_t(a, b, t1, t2, r1, r2):
    '''
    x1 = np.sum(a ** 2,axis=-1,keepdims=True) + np.sum(b ** 2,axis=-1,keepdims=True).T
    x3 = np.matmul(r1**2, (t1**2).T) + np.matmul(r2**2, (t2**2).T) -2*np.matmul(r1*r2,(t1*t2).T)
    x4 = -2*np.matmul(a,b.T)
    x5 = 2*np.matmul(a*r1, t1.T) - 2*np.matmul(a*r2,t2.T) - 2*np.matmul(r1, (b*t1).T) + 2*np.matmul(r2,(b*t2).T)
    '''
    x1 = torch.sum(a * a, dim=-1, keepdim=True) + torch.sum(b * b, dim=-1, keepdim=True).t()
    x2 = (r1 * r1) @ (t1 * t1).t() + (r2 * r2) @ (t2 * t2).t() - 2.0 * (r1 * r2) @ (t1 * t2).t()
    x3 = -2.0 * (a @ b.t())
    x4 = 2.0 * ((a * r1) @ t1.t()) - 2.0 * ((a * r2) @ t2.t()) - 2.0 * (r1 @ (b * t1).t()) + 2.0 * (r2 @ (b * t2).t())
    print(x1[0,0],x2[0,0],x3[0,0],x4[0,0])
    return x1+x2+x3+x4

def norm(x):
    givens = x.view((x.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    return givens.view((x.shape[0],-1))

res = euc_sqdistance_t(torch.tensor(a), torch.tensor(b),
                       torch.tensor(t1), torch.tensor(t2),
                       torch.tensor(r1), torch.tensor(r2))

print(res[0,0].numpy())
'''
print(torch.tensor([1,2,3,4,5,6])[::2])
print(torch.tensor([1,2,3,4,5,6])[1::2])
'''

a = torch.tensor(np.ones([N1, d]) * 0.0)
b = torch.tensor(np.ones([N2, d]) * 0.0)
t = torch.tensor(np.random.random([N2, d]))
r = torch.tensor(np.random.random([N1, d]))

rn = norm(r)

res0 = - euc_sqdistance_t(a[:,0::2],b[:,0::2], t[:,0::2], t[:,1::2],rn[:,0::2],rn[:,1::2])
res1 = - euc_sqdistance_t(a[:,1::2],b[:,1::2], t[:,1::2],-t[:,0::2],rn[:,0::2],rn[:,1::2])

print(res0[0,0],res1[0,0])

a = a.view([N1,1,d])
b = b.view([1,N2,d])
#t = t.view([1,N2,d])
#r = r.view([N1,1,d])
#t = t.repeat(N1, 1,1).view([-1,d])
#r = r.repeat(1, N2,1).view([-1,d])
tr = givens_rotations(r[0:1],t[0:1])
#res2 = torch.sum(((a-b+tr)**2)[:,:,0::2],dim=-1)
#res3 = torch.sum(((a-b+tr)**2)[:,:,1::2],dim=-1)
r1 = r[0:1,0::2]
r2 = r[0:1,1::2]
t1 = t[0:1,0::2]
t2 = t[0:1,1::2]
print(torch.sum((t1*r1-t2*r2)**2,dim=-1))
res2 = torch.sum((tr**2)[:,0::2],dim=-1)
res3 = torch.sum((tr**2)[:,1::2],dim=-1)
print(-res2,-res3)