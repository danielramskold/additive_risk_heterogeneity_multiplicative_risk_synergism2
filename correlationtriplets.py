#! /usr/bin/python3
import argparse, pandas, numpy, math, random
from scipy import stats
from sklearn import linear_model

# derived from use_linlee-logprism_v5.py and use_linlee-logprism_v6.py

def OR(df, colname, maxincol, reraise_error=False):
    dfS = df[df['outcome']==True]
    dfH = df[df['outcome']==False]
    try:
        if colname in ('A', 'B'):
            return dfS[colname].sum() / dfH[colname].sum() / abs(maxincol-dfS[colname]).sum() * abs(maxincol-dfH[colname]).sum()
        else:
            return dfS[colname].sum() / dfH[colname].sum() / dfS['A0B0'].sum() * dfH['A0B0'].sum()
    except ZeroDivisionError:
        if reraise_error: raise
        return float('nan')

def state_from_two(model, bases_p, yes_val):
    if model == 'dom': # yes_val is dominant
        return [(v1==yes_val or v2==yes_val) if v1 != '0' and v2 != '0' else None for v1,v2 in zip(bases_p[::2], bases_p[1::2])]
    elif model == 'rec': # yes_val is recessive
        return [(v1==yes_val and v2==yes_val) if v1 != '0' and v2 != '0' else None for v1,v2 in zip(bases_p[::2], bases_p[1::2])]
    elif model == '1al': # pick one allele, alternate which one
        return [V[i%2]==yes_val if V[i%2] != '0' else None for i, V in enumerate(zip(bases_p[::2], bases_p[1::2]))]
    elif model == '1st': # pick one allele
        return [V[0]==yes_val if V[0] != '0' else None for i, V in enumerate(zip(bases_p[::2], bases_p[1::2]))]
    elif model == '1rand': # pick one allele
        return [random.choice(V)==yes_val if V[0] != '0' and V[1] != '0' else None for i, V in enumerate(zip(bases_p[::2], bases_p[1::2]))]
    elif model == 'sum': # allelic, a form of co-dominant
        return [(int(v1==yes_val) + int(v2==yes_val)) if v1 != '0' and v2 != '0' else None for v1,v2 in zip(bases_p[::2], bases_p[1::2])]
# add allelic model: counts of alleles (codominant, which is how SE is)


def residual_of_linear_fit(data_column1, data_control_variable):
    fit1 = linear_model.LinearRegression(fit_intercept=True)
    fit1.fit(data_control_variable, data_column1)
    return data_column1 - (numpy.dot(data_control_variable, fit1.coef_) + fit1.intercept_)

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('tfam')
    parser.add_argument('tped12')
    parser.add_argument('SE_tfam')
    parser.add_argument('SE_tpedYN')
    parser.add_argument('--rand', choices=['samegroupLDpres', 'midfreqsepgroups', 'bgfreqsepgroups', 'bgfreqsepgroupsv2', 'samegroup'])
    parser.add_argument('--model', choices=['dom', 'rec', '1al', '1st', '1rand', 'sum'], required=True)
    parser.add_argument('--SEmodel', choices=['dom', 'rec', '1al', '1st', '1rand', 'sum'])
    parser.add_argument('--bootstrap_replicate', '--bootstrap', action='store_true', dest='bootstrap_replicate')
    parser.add_argument('--covariates')
    parser.add_argument('--shuffle_covariates', action='store_true')
    o = parser.parse_args()
    o.risk_as_1 = True
    
    if o.SEmodel is None: o.SEmodel = o.model
    maxA = 2 if o.SEmodel == 'sum' else 1
    maxB = 2 if o.model == 'sum' else 1
    maxAB = maxA * maxB
    
    
    
    df_master = pandas.DataFrame()
    
    casecontol = []
    SE_samplelist = []
    with open(o.SE_tfam, 'r') as infh:
        for li, line in enumerate(infh):
            p = line.split()
            casecontol.append(None if p[5] not in '12' else (p[5] == '2'))
            SE_samplelist.append(p[0])
    
    df_master['outcome'] = pandas.Series(casecontol, index=SE_samplelist).astype(bool)
    
    with open(o.SE_tpedYN, 'r') as infh:
        for line in infh:
            p = line.split()
            bases_p = p[4:]
            df_master['A'] = state_from_two(o.SEmodel, bases_p, 'Y')
    
    if o.covariates is not None:
        covar_table = pandas.read_csv(o.covariates, sep='\t', index_col=0).reindex(df_master.index)
        if o.shuffle_covariates:
            covar_table = covar_table.sample(frac=1)
            covar_table.index = df_master.index
        empty_covar_rows = covar_table.isnull().any(1)
        if any(empty_covar_rows):
            print(covar_table.index[empty_covar_rows])
            print('Empty row in covariate table')
            exit()
    
    if o.rand == 'samegroupLDpres' or o.rand == 'samegroup':
        df_master.loc[df_master['outcome']==True, 'A'] = list(df_master.loc[df_master['outcome']==True, 'A'].sample(frac=1).reset_index(drop=True))
        df_master.loc[df_master['outcome']==False, 'A'] = list(df_master.loc[df_master['outcome']==False, 'A'].sample(frac=1).reset_index(drop=True))
    elif o.rand == 'midfreqsepgroups':
        #split into two groups, with equal fraction of cases between them. in one, shuffle X, in the other shuffle Y.   This will be the mid-freq sep-group model.
        # odds ratio for A will drop a bit unfortunately
        df_master['group'] = numpy.random.choice(2, df_master.shape[0])
        df_master.loc[df_master['group']==0, 'A'] = list(df_master.loc[df_master['group']==0, 'A'].sample(frac=1).reset_index(drop=True))
    elif o.rand == 'bgfreqsepgroups':
        # split into two groups, with equal fraction of cases between them. in one, shuffle controls for X and spike in X into cases at same frquency as in controls, e.g. by randomly drawing from controls. same for Y in the other.    This will be the low-freq sep-group model.
        df_master['group'] = numpy.random.choice(2, df_master.shape[0])
        df_master.loc[df_master['group']==0, 'A'] = list(df_master.loc[(df_master['group']==0) & (df_master['outcome']==False), 'A'].sample(n=sum(df_master['group']==0), replace=True).reset_index(drop=True))
    elif o.rand == 'bgfreqsepgroupsv2':
        # try to deal with the too-high variance
        df_master['group'] = numpy.random.choice(2, df_master.shape[0])
        df_master.loc[(df_master['group']==0) & (df_master['outcome']==True), 'A'] = list(df_master.loc[(df_master['group']==0) & (df_master['outcome']==False), 'A'].sample(n=sum((df_master['group']==0) & (df_master['outcome']==True)), replace=True).reset_index(drop=True))
    
    SNPs_samplelist = []
    with open(o.tfam, 'r') as infh:
        for li, line in enumerate(infh):
            p = line.split()
            SNPs_samplelist.append(p[0])
    
    df_master = df_master.reindex(SNPs_samplelist)
    
    outputs = 'SNP', 'r_all', 'r_case', 'r_ctrl', 'OR_A', 'OR_B', 'P_all', 'P_case', 'P_ctrl'
    print('\t'.join(list(map(str, outputs))))
    
    with open(o.tped12, 'r') as infh:
        for line in infh:
            p = line.split()
            bases_p = p[4:]
            df_master['B'] = state_from_two(o.model, bases_p, '1')
            df = df_master.dropna().copy()
            try:
                if o.risk_as_1 and OR(df, 'B', maxB, True) < 1:
                    df['B'] = maxB - df['B'].astype(int)     # should this be here or after the shuffle/bootstrap steps? would give some srtange effects, esp OR confidence interval, if it was placed just before df['A1B1'] = df['A'] * df['B'] I figure
            except ZeroDivisionError:
                continue
            
            if o.rand == 'samegroup':
                df.loc[df['outcome']==True, 'B'] = list(df.loc[df['outcome']==True, 'B'].sample(frac=1).reset_index(drop=True))
                df.loc[df['outcome']==False, 'B'] = list(df.loc[df['outcome']==False, 'B'].sample(frac=1).reset_index(drop=True))
            elif o.rand == 'midfreqsepgroups':
                df.loc[df['group']==1, 'B'] = list(df.loc[df['group']==1, 'B'].sample(frac=1).reset_index(drop=True))
            elif o.rand == 'bgfreqsepgroups':
                df.loc[df['group']==1, 'B'] = list(df.loc[(df['group']==1) & (df['outcome']==False), 'B'].sample(n=sum(df['group']==1), replace=True).reset_index(drop=True))
            elif o.rand == 'bgfreqsepgroupsv2':
                df.loc[(df['group']==1) & (df['outcome']==True), 'B'] = list(df.loc[(df_master['group']==1) & (df['outcome']==False), 'B'].sample(n=sum((df['group']==1) & (df['outcome']==True)), replace=True).reset_index(drop=True))
            
            if o.bootstrap_replicate:
                df = df.sample(frac=1, axis=0, replace=True).reset_index(drop=True)    # not setting random_state, thus breaking LD
            
            df['A'] = df['A'].astype(int)
            df['B'] = df['B'].astype(int)
            
            if o.covariates is not None:
                # partial correlation
                covar_table_same_rows = covar_table.loc[df.index, :]
                residuals = pandas.DataFrame({'A':residual_of_linear_fit(df['A'], covar_table_same_rows), 'B':residual_of_linear_fit(df['B'], covar_table_same_rows)}, index=df.index)
                residualsS = residuals[df['outcome']==True]
                residualsH = residuals[df['outcome']==False]
                r_all, P_all = stats.pearsonr(residuals['A'], residuals['B'])
                r_case, P_case = stats.pearsonr(residualsS['A'], residualsS['B'])
                r_ctrl, P_ctrl = stats.pearsonr(residualsH['A'], residualsH['B'])
            else:
                dfS = df[df['outcome']==True]
                dfH = df[df['outcome']==False]
                r_all, P_all = stats.pearsonr(df['A'], df['B'])
                r_case, P_case = stats.pearsonr(dfS['A'], dfS['B'])
                r_ctrl, P_ctrl = stats.pearsonr(dfH['A'], dfH['B'])
            
            try:
                outputs = p[1], r_all, r_case, r_ctrl, OR(df, 'A', maxA), OR(df, 'B', maxB), P_all, P_case, P_ctrl
                print('\t'.join(list(map(str, outputs))))
                
            except BrokenPipeError:
                break
