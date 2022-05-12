#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lefteris

@subject: supply-demand optimization class
"""


import pandas as pd
import numpy as np
import pulp


class SupplyDemand:
    def __init__(self,C, D,S,c):
        #All arrays
        self.C = C #transportation cost
        self.D = D #demand cost
        self.S = S #supply cost
        self.c = c #oporational cost
        self.nVar = c.shape[0]
        self.mVar = D.shape[0]
    
        
    def get_variables(self):
        xshape = (range(self.nVar), range(self.mVar))
        x = pulp.LpVariable.dicts("X", xshape,lowBound = 0,  cat='Continuous')
        y = pulp.LpVariable.dicts("Y", range(self.nVar), cat=pulp.LpBinary)
        return x, y

    def get_solution(self):
        x, y = self.get_variables()
        
        prob = pulp.LpProblem("distribution_opt", pulp.LpMinimize)
        
        objective_function = sum([ self.C[0,m_idx] * x[0][m_idx] for m_idx in range(self.mVar)]) + self.c[0] * y[0]
        for n_idx in range(1,self.nVar):
            objective_function += sum([ self.C[n_idx,m_idx] * x[n_idx][m_idx] for m_idx in range(self.mVar)]) + self.c[n_idx] * y[n_idx]
        
        prob += objective_function
        
        #Constraints
        for n_idx in range(self.nVar):
            prob += sum(x[n_idx][m_idx] for m_idx in range(self.mVar)) <= self.S[n_idx] * y[n_idx]
        for m_idx in range(self.mVar):
            prob += sum(x[n_idx][m_idx] for n_idx in range(self.nVar)) == self.D[m_idx]
    
    
        prob.solve(pulp.apis.PULP_CBC_CMD(msg=False))
        return prob
    def get_network(self):
        prob = self.get_solution()
        X = {'pl_id' : [],
             'wr_id' : [],
            'value' : []}
        Y = {'pl_id': [],
             'value' : []}
        for v in prob.variables():
            ids = v.name[2:].split("_")
            if v.name[0] == 'X':
                X['pl_id'].append(int(ids[0]))
                X['wr_id'].append(int(ids[1]))
                X['value'].append(v.varValue)
            else:
                Y['pl_id'].append(int(ids[0]))
                Y['value'].append(v.varValue)
        
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        Xfilt = X.loc[X['value']>0].reset_index(drop=True)
        return(prob, Xfilt, Y)
    def get_report(self):
        prob, X, Y = self.get_network()

        X = X.rename(columns = {'value': 'supply'})
        C_df = pd.DataFrame(self.C)
        C_df['pl_id'] = range(self.C.shape[0])
        C_df = pd.melt(C_df, id_vars=['pl_id'])
        C_df = C_df.rename(columns = {'variable': 'wr_id',
                                      'value': 'transport_cost'})

        C_df['wr_id'] = C_df['wr_id'].astype('int') 


        D_df = pd.DataFrame(
                {'wr_id': range(self.D.shape[0]),
                 'demand': self.D}
            )

        S_df = pd.DataFrame(
                {'pl_id': range(self.S.shape[0]),
                 'limit_supply': self.S}
            )

        c_df = pd.DataFrame(
                {'pl_id': range(self.c.shape[0]),
                 'operate_cost': self.c}
            )


        report = X.merge(C_df, left_on=['pl_id','wr_id'], right_on=['pl_id','wr_id'])

        report = report.merge(S_df, left_on='pl_id', right_on='pl_id')
        report = report.merge(D_df, left_on='wr_id', right_on='wr_id')
        report = report.merge(c_df, left_on='pl_id', right_on='pl_id')
        return report




if __name__ == "__main__":
    df = pd.read_csv("data/demand.csv", sep='\s+')
    C = np.array(df.iloc[:-1,1:-2])
    D = np.array(df.iloc[-1,1:-2])
    S = np.array(df.iloc[:-1,-2])
    c = np.array(df.iloc[:-1,-1])
    df
    
    supp = SupplyDemand(C,D,S,c)
    prob, X, Y = supp.get_network()
    print(prob)
    print(X)
    