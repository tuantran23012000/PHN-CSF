import torch
from tools.min_norm_solvers_numpy import MinNormSolver
import numpy as np
class CS_functions():
    def __init__(self,losses,ray):
        super().__init__()
        self.losses = losses
        self.ray = ray
    def linear_function(self):
        ls = (self.losses * self.ray).sum()
        return ls

    def log_function(self):
        return (self.ray*torch.log(self.losses+1)).sum()

    def ac_function(self,rho):
        ls = (self.losses * self.ray).sum()
        cheby = max(self.losses * self.ray)
        return cheby + rho*ls

    def mc_function(self,rho):
        ls = (self.losses * self.ray).sum()
        cheby = max(self.losses * self.ray + rho*ls)
        return cheby
    
    def hv_function(self,dynamic_weight,rho):
        rl = self.losses * self.ray
        l_s = torch.norm(self.losses)
        r_s = torch.norm(self.ray)
        cosine = - (rl.sum()) / (l_s*r_s) 
        hv = -(dynamic_weight*self.losses).sum()  + rho * cosine
        return hv

    def product_function(self):
        return torch.prod((self.losses+1)**self.ray)

    def cosine_function(self):
        rl =self.losses * self.ray
        l_s = torch.sqrt((self.losses**2).sum())
        r_s = torch.sqrt((self.ray**2).sum())
        cosine = - (rl.sum()) / (l_s*r_s)
        return cosine

    def utility_function(self,ub):
        
        U = 1/torch.prod((ub - self.losses)**self.ray)
        return U

    def chebyshev_function(self):
        
        cheby = max(self.losses * self.ray)
        
        return cheby

    def KL_function(self):
        m = len(self.losses)
        rl = torch.exp(self.losses * self.ray)
        normalized_rl = rl / (rl.sum())
        KL = (normalized_rl * torch.log(normalized_rl * m)).sum() 
        return KL

    def cauchy_schwarz_function(self):
        rl = self.losses * self.ray
        l_s = (self.losses**2).sum()
        r_s = (self.ray**2).sum()
        cauchy_schwarz = 1 - ((rl.sum())**2 / (l_s*r_s))
        return cauchy_schwarz
    
    def get_d_paretomtl(self,grads,value,normalized_rest_weights,normalized_current_weight):
        # calculate the gradient direction for Pareto MTL
        nobj, dim = grads.shape
        
        # check active constraints
        #normalized_current_weight = weights[i]/np.linalg.norm(weights[i])
        #normalized_rest_weights = np.delete(weights, (i), axis=0) / np.linalg.norm(np.delete(weights, (i), axis=0), axis = 1,keepdims = True)
        w = normalized_rest_weights - normalized_current_weight
        
        #w = normalized_rest_weights
        # solve QP 
        gx =  torch.matmul(w,value/torch.norm(value))
        idx = gx >  0
        if torch.sum(idx) <= 0:
            sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
            #print(sol)
            return (torch.tensor(sol).float()* value).sum()
        vec =  torch.cat((grads, torch.matmul(w[idx],grads)))
        
        #    # use cvxopt to solve QP
        #    
        #    P = np.dot(vec , vec.T)
        #     
        #    q = np.zeros(nobj + np.sum(idx))
        #    
        #    G =  - np.eye(nobj + np.sum(idx) )
        #    h = np.zeros(nobj + np.sum(idx))
        #    
        #
        #    
        #    A = np.ones(nobj + np.sum(idx)).reshape(1,nobj + np.sum(idx))
        #    b = np.ones(1)
        
        #    cvxopt.solvers.options['show_progress'] = False
        #    sol = cvxopt_solve_qp(P, q, G, h, A, b)
    
        # use MinNormSolver to solve QP
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])
        
        # reformulate ParetoMTL as linear scalarization method, return the weights
        weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2,0] for j in torch.arange(2,2 + torch.sum(idx))]))
        weight1 = sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2,1] for j in torch.arange(2,2 + torch.sum(idx))]))
        weight = torch.stack([weight0,weight1])
        # weight += weight*(2/torch.sum(weight))
        # print(weight)
        # print(value)
        return (weight * value).sum()
from abc import abstractmethod

import cvxopt
import cvxpy as cp
import numpy as np
import torch
class Solver:
    def __init__(self):
        super().__init__()
        # self.n_tasks = n_tasks

    @abstractmethod
    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        pass

    def __call__(self, losses, ray, parameters, **kwargs):
        return self.get_weighted_loss(losses, ray, parameters, **kwargs)    
class EPOSolver(Solver):
    """Wrapper over EPO"""

    def __init__(self, n_tasks, n_params):
        super().__init__()
        self.solver = EPO(n_tasks=n_tasks, n_params=n_params)

    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        assert parameters is not None
        return self.solver.get_weighted_loss(losses, ray, parameters)


class EPO:
    def __init__(self, n_tasks, n_params):
        self.n_tasks = n_tasks
        self.n_params = n_params

    def __call__(self, losses, ray, parameters):
        return self.get_weighted_loss(losses, ray, parameters)

    @staticmethod
    def _flattening(grad):
        return torch.cat(
            tuple(
                g.reshape(
                    -1,
                )
                for i, g in enumerate(grad)
            ),
            axis=0,
        )

    def get_weighted_loss(self, losses, ray, parameters):
        lp = ExactParetoLP(m=self.n_tasks, n=self.n_params, r=ray.cpu().numpy())

        grads = []
        for i, loss in enumerate(losses):
            g = torch.autograd.grad(loss, parameters, retain_graph=True)
            flat_grad = self._flattening(g)
            grads.append(flat_grad.data)

        G = torch.stack(grads)
        GG_T = G @ G.T
        GG_T = GG_T.detach().cpu().numpy()

        numpy_losses = losses.detach().cpu().numpy()

        try:
            alpha = lp.get_alpha(numpy_losses, G=GG_T, C=True)
        except Exception as excep:
            #print("Ex: ",excep)
            alpha = None

        if alpha is None:  # A patch for the issue in cvxpy
            alpha = (ray / ray.sum()).cpu().numpy()

        alpha *= self.n_tasks
        alpha = torch.from_numpy(alpha).to(losses.device)

        weighted_loss = torch.sum(losses * alpha)
        return weighted_loss


class ExactParetoLP(object):
    """modifications of the code in https://github.com/dbmptr/EPOSearch"""

    def __init__(self, m, n, r, eps=1e-4):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.n = n
        self.r = r
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)  # Adjustments
        self.C = cp.Parameter((m, m))  # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)  # d_bal^TG
        self.rhs = cp.Parameter(m)  # RHS of constraints for balancing

        self.alpha = cp.Variable(m)  # Variable to optimize

        obj_bal = cp.Maximize(self.alpha @ self.Ca)  # objective for balance
        constraints_bal = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,  # Simplex
            self.C @ self.alpha >= self.rhs,
        ]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,  # Restrict
            self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
            self.C @ self.alpha >= 0,
        ]
        constraints_rel = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,  # Relaxed
            self.C @ self.alpha >= 0,
        ]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance

        self.gamma = 0  # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0  # Stores the latest non-uniformity

    def get_alpha(self, l, G, r=None, C=False, relax=False):
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"
        rl, self.mu_rl, self.a.value = adjustments(l, r)
        self.C.value = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        if self.mu_rl > self.eps:
            J = self.Ca.value > 0
            if len(np.where(J)[0]) > 0:
                J_star_idx = np.where(rl == np.max(rl))[0]
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf  # Not efficient; but works.
                self.rhs.value[J_star_idx] = 0
            else:
                self.rhs.value = np.zeros_like(self.Ca.value)
            self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "bal"
        else:
            if relax:
                self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
            else:
                self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "dom"

        return self.alpha.value


def mu(rl, normed=False):
    if len(np.where(rl < 0)[0]):
        raise ValueError(f"rl<0 \n rl={rl}")
        return None
    m = len(rl)
    l_hat = rl if normed else rl / rl.sum()
    eps = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]
    return np.sum(l_hat * np.log(l_hat * m))


def adjustments(l, r=1):
    m = len(l)
    rl = r * l
    l_hat = rl / rl.sum()
    mu_rl = mu(l_hat, normed=True)
    a = r * (np.log(l_hat * m) - mu_rl)
    return rl, mu_rl, a