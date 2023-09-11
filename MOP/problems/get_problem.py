import numpy as np
import torch
from matplotlib import pyplot as plt
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
def get_ref_dirs(n_obj):
    if n_obj == 2:
        ref_dirs = UniformReferenceDirectionFactory(2, n_points=100).do()
    elif n_obj == 3:
        #ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=15).do()
        ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=100).do()
    else:
        raise Exception("Please provide reference directions for more than 3 objectives!")
    return ref_dirs
def generic_sphere(ref_dirs):
    return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))
class DTLZ2():
    def __init__(self):
        self.a = 1
    def create_pf(self):
        ref_dirs = get_ref_dirs(3)
        #print(ref_dirs.shape)
        pf =  generic_sphere(ref_dirs)
        return pf
    def f_1(self, output):
        return (torch.cos(torch.pi/2*output[0, 0])*torch.cos(torch.pi/2*output[0, 1])*(sum((output[0, 2:]-0.5)**2)+1))
    def f_2(self, output):
        return (torch.cos(torch.pi/2*output[0, 0]**self.a)*torch.sin(torch.pi/2*output[0, 1]**self.a)*(sum((output[0, 2:]-0.5)**2)+1))
    def f_3(self, output):
        return torch.sin(torch.pi/2*output[0, 0]**self.a)*(sum((output[0, 2:]-0.5)**2)+1)
class test():
    def __init__(self):
        #self.a = 1
        self.num = 1000
    def create_pf(self):
        ps = np.linspace(-10,10,num = self.num)
        pf = []
        for x1 in ps:
            for x2 in ps:
                x = torch.Tensor([[x1,x2]])
                f= np.stack([self.f_1(x).item(),self.f_2(x).item()])
                pf.append(f)   
        pf = np.array(pf)
        return pf
    def f_1(self, output):
        return torch.cos(output[0, 0])**2 + 0.2
    def f_2(self, output):
        return 1.3+torch.sin(output[0, 1])**2- torch.cos(output[0, 0])-0.1*torch.sin(22*torch.pi*torch.cos(output[0, 0])**2)**5
class ex1():
    def __init__(self):
        self.num = 1000
    def create_pf(self):
        ps = np.linspace(0,1,num = self.num)
        pf = []
        for x1 in ps:
            x = torch.Tensor([[x1]])
            f= np.stack([self.f_1(x).item(),self.f_2(x).item()])
            pf.append(f)   
        pf = np.array(pf)
        return pf
    def f_1(self, output):
        return output[:,0]
    def f_2(self, output):
        return (output[:,0]-1)**2
class ex2():
    def __init__(self):
        self.num = 1000
    def create_pf(self):
        ps = np.linspace(0,5,num = self.num)
        pf = []
        for x1 in ps:
            x = torch.Tensor([[x1,x1]])
            f= np.stack([self.f_1(x).item(),self.f_2(x).item()])
            pf.append(f)   
        pf = np.array(pf)
        return pf
    def f_1(self, output):
        return (1/50)*(output[0][0]**2 + output[0][1]**2)
    def f_2(self, output):
        return (1/50)*((output[0][0]-5)**2 + (output[0][1]-5)**2)
class ex3():
    def __init__(self):
        self.num = 50
    def create_pf(self):
        u = np.linspace(0, 1, endpoint=True, num=self.num)
        v = np.linspace(0, 1, endpoint=True, num=self.num)
        tmp = []
        for i in u:
            for j in v:
                if 1-i**2-j**2 >=0:
                    tmp.append([np.sqrt(1-i**2-j**2),i,j])
                    tmp.append([i,np.sqrt(1-i**2-j**2),j])
                    tmp.append([i,j,np.sqrt(1-i**2-j**2)])
        uv = np.array(tmp)
        print(f"uv.shape={uv.shape}")
        ls = []
        for x in uv:
            x = torch.Tensor([x])
            f= np.stack([self.f_1(x).item(),self.f_2(x).item(),self.f_3(x).item()])
            ls.append(f)
        ls = np.stack(ls)
        po, pf = [], []
        for i, x in enumerate(uv):
            l_i = ls[i]
            po.append(x)
            pf.append(l_i)
        po = np.stack(po)
        pf = np.stack(pf)
        return pf
    def f_1(self, output):
        return ((output[0][0]**2 + output[0][1]**2 + output[0][2]**2+output[0][1] - 12*(output[0][2])) +12)/14

    def f_2(self, output):
        return ((output[0][0]**2 + output[0][1]**2 + output[0][2]**2\
            + 8*(output[0][0]) - 44.8*(output[0][1]) + 8*(output[0][2])) +44)/57
    def f_3(self, output):
        return ((output[0][0]**2 + output[0][1]**2 + output[0][2]**2 -44.8*(output[0][0])\
            + 8*(output[0][1]) + 8*(output[0][2]))+43.7)/56
class ex4():
    def __init__(self):
        self.num = 1000
    def create_pf(self):
        pf = [[0,0]]
        pf = np.array(pf)
        return pf
    def f_1(self, output):
        return output[0][0]
    def f_2(self, output):
        return output[0][1]
class ZDT1():
    def __init__(self):
        self.n_pareto_points = 1000
    def create_pf(self):
        x = np.linspace(0, 1, self.n_pareto_points)
        pf = np.array([x, 1 - np.sqrt(x)]).T
        return pf
    def f_1(self, output):
        return output[0][0]
    def f_2(self, output):
        dim = output.shape[1]
        tmp = 0
        for i in range(1,dim):
            tmp += output[0][i]
        g = 1 + (9/(dim-1))*tmp
        f1 = output[0][0]
        return g*(1 - torch.sqrt(f1/g))
class ZDT2():
    def __init__(self):
        self.n_pareto_points = 1000
    def create_pf(self):
        x = np.linspace(0, 1, self.n_pareto_points)
        pf = np.array([x, 1 - (x)**2]).T
        return pf
    def f_1(self, output):
        return output[0][0]
    def f_2(self, output):
        dim = output.shape[1]
        tmp = 0
        for i in range(1,dim):
            tmp += output[0][i]
        g = 1 + (9/(dim-1))*tmp
        f1 = output[0][0]
        return g*(1 - (f1/g)**2)
class ZDT3():
    def __init__(self):
        self.n_points = 1000 
    def create_pf(self):
        regions = [[0, 0.0830015349],
                   [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041],
                   [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]

        pf = []
        flatten = True
        for r in regions:
            x1 = np.linspace(r[0], r[1], int(self.n_points / len(regions)))
            x2 = 1 + (1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1))
            pf.append(np.array([x1, x2]).T)

        if not flatten:
            pf = np.concatenate([pf[None,...] for pf in pf])
        else:
            pf = np.row_stack(pf)

        return pf
    def f_1(self, output):
        return output[0][0]
    def f_2(self, output):
        dim = output.shape[1]
        tmp = 0
        for i in range(1,dim):
            tmp += output[0][i]
        g = 1 + (9/(dim-1))*tmp
        f1 = output[0][0]
        return 1 + g*(1 - torch.sqrt(f1/g) - (f1/g)*torch.mean(torch.sin(10*torch.pi*f1)))
class Problem():
    def __init__(self,name, mode):
        self.name = name
        if self.name == 'ex1':
            self.pb = ex1()
        elif self.name == 'ex2':
            self.pb = ex2()
        elif self.name == 'ex3':
            self.pb = ex3()
        elif self.name == 'ex4':
            self.pb = ex4()
        elif self.name == 'ZDT1':
            self.pb = ZDT1()
        elif self.name == 'ZDT2':
            self.pb = ZDT2()
        elif self.name == 'ZDT3':
            self.pb = ZDT3()
        elif self.name == 'DTLZ2':
            self.pb = DTLZ2()
        elif self.name == 'test':
            self.pb = test()
        self.mode = mode
    def get_pf(self):
        pf = self.pb.create_pf()
        return pf
    def get_values(self, output):
        if self.mode == '2d':
            f1, f2 = self.pb.f_1(output), self.pb.f_2(output)
            objectives = [f1, f2]
        else:
            f1, f2, f3 = self.pb.f_1(output), self.pb.f_2(output), self.pb.f_3(output)
            objectives = [f1, f2, f3]
        return objectives

