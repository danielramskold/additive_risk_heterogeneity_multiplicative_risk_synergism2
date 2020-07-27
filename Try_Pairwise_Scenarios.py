#!/usr/bin/env python3
import pandas, random, math, argparse, functools

def same_cause_sim_modelI(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence, locus_modelA, locus_modelB):
    df = pandas.DataFrame()
    case_control_split(df, 'outcome', prevalence, n)
    spike_in(df, 'outcome', 'A', a_rel, bg_freq1, locus_modelA)
    spike_in(df, 'outcome', 'B', b_rel, bg_freq2, locus_modelB)
    return df

def OR_logic_sim_modelV(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence, locus_modelA, locus_modelB):
    df = pandas.DataFrame()
    case_control_split(df, 'a_cases', 1-math.sqrt(1-prevalence), n)
    case_control_split(df, 'b_cases', 1-math.sqrt(1-prevalence), n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1, locus_modelA)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2, locus_modelB)
    df['outcome'] = df['a_cases'] | df['b_cases']
    return df

def AND_logic_sim_modelIV(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence, locus_modelA, locus_modelB):
    df = pandas.DataFrame()
    case_control_split(df, 'a_cases', math.sqrt(prevalence), n)
    case_control_split(df, 'b_cases', math.sqrt(prevalence), n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1, locus_modelA)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2, locus_modelB)
    df['outcome'] = df['a_cases'] & df['b_cases']
    return df

def separate_cause_sim_modelIII(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence, locus_modelA, locus_modelB):
    df_a = pandas.DataFrame()
    df_b = pandas.DataFrame()
    case_control_split(df_a, 'outcome', prevalence, n//2)
    case_control_split(df_b, 'outcome', prevalence, n-n//2)
    spike_in(df_a, 'outcome', 'A', a_rel, bg_freq1, locus_modelA)
    spike_in(df_b, 'outcome', 'A', 1, bg_freq1, locus_modelB)
    spike_in(df_a, 'outcome', 'B', 1, bg_freq2, locus_modelA)
    spike_in(df_b, 'outcome', 'B', b_rel, bg_freq2, locus_modelB)
    df = pandas.concat((df_a, df_b), ignore_index=True)
    return df

def separate2_cause_sim_modelII(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence, locus_modelA, locus_modelB):
    df = pandas.DataFrame()
    case_control_split(df, 'outcome', prevalence, n)
    spike_in(df, 'outcome', 'A', a_rel, bg_freq1, locus_modelA)
    spike_in(df, 'outcome', 'B', b_rel, bg_freq2, locus_modelB)
    case_control_split(df, 'group', 0.5, n)
    shuffle(df, 'A', df['group'])
    shuffle(df, 'B', ~df['group'])
    return df

def two_group_with_different_prevalence(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence, locus_modelA, locus_modelB):
    x1 = bg_freq1
    x2 = bg_freq1*a_rel
    y1 = bg_freq2
    y2 = bg_freq2*b_rel
    prevalence_ratio = 2
    prev1 = 2 * prevalence / (prevalence_ratio + 1)
    prev2 = 2 * prevalence / (1/prevalence_ratio + 1)
    df_a = pandas.DataFrame()
    df_b = pandas.DataFrame()
    case_control_split(df_a, 'outcome', prev1, n//2)
    case_control_split(df_b, 'outcome', prev2, n-n//2)
    spike_in(df_a, 'outcome', 'A', 1, x1, locus_modelA)
    spike_in(df_b, 'outcome', 'A', 1, x2, locus_modelA)
    spike_in(df_a, 'outcome', 'B', 1, y1, locus_modelB)
    spike_in(df_b, 'outcome', 'B', 1, y2, locus_modelB)
    df = pandas.concat((df_a, df_b), ignore_index=True)
    return df

def threshold_three_of_five(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence, locus_modelA, locus_modelB, threshold):
    df = pandas.DataFrame()
    subcasefreq = -math.log(threshold+prevalence, prevalence)   # not really based on anything, but sometimes it comes close ...and sometimes not
    
    if prevalence == 0.005:
        if threshold==1: subcasefreq = 0.001  # 0.005/5 - overlap~o(prev^2)
        elif threshold==2: subcasefreq = 0.022
        elif threshold==3: subcasefreq = 0.085
        elif threshold==4: subcasefreq = 0.184
        elif threshold==5: subcasefreq = 0.005**0.2 # 0.3465724215775732
    elif prevalence == 0.015:
        if threshold==1: subcasefreq = 0.0029
        elif threshold==2: subcasefreq = 0.04
        elif threshold==3: subcasefreq = 0.12
        elif threshold==4: subcasefreq = 0.25
        elif threshold==5: subcasefreq = 0.015**0.2  # 0.43173598837665544
    elif prevalence == 0.05:
        if threshold==1: subcasefreq = 0.0097
        elif threshold==2: subcasefreq = 0.078
        elif threshold==3: subcasefreq = 0.19
        elif threshold==4: subcasefreq = 0.34
        elif threshold==5: subcasefreq = 0.05**0.2 # 0.5492802716530588
    elif prevalence == 0.15:
        if threshold==1: subcasefreq = 0.033
        elif threshold==2: subcasefreq = 0.14
        elif threshold==3: subcasefreq = 0.29
        elif threshold==4: subcasefreq = 0.47
        elif threshold==5: subcasefreq = 0.15**0.2  # 0.6842554289186317
    elif prevalence == 0.5:
        if threshold==1: subcasefreq = 0.13
        elif threshold==2: subcasefreq = 0.32
        elif threshold==3: subcasefreq = 0.5
        elif threshold==4: subcasefreq = 0.7
        elif threshold==5: subcasefreq = 0.5**0.2 # 0.8705505632961241
    else:
        print('Please use a prevalence of 0.5, 0.15, 0.05, 0.015 or 0.005 for this 5-factor implementation of the multifactorial threshold model')
        exit(1)
    
    
    case_control_split(df, 'a_cases', subcasefreq, n)
    case_control_split(df, 'b_cases', subcasefreq, n)
    case_control_split(df, 'c_cases', subcasefreq, n)
    case_control_split(df, 'd_cases', subcasefreq, n)
    case_control_split(df, 'e_cases', subcasefreq, n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1, locus_modelA)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2, locus_modelB)
    df['outcome'] = df['a_cases'].astype(int) + df['b_cases'].astype(int) + df['c_cases'].astype(int) + df['d_cases'].astype(int) + df['e_cases'].astype(int) >= threshold
    return df

def spike_in(df, case_col, new_col, rel, base_freq, locus_model):
    if locus_model == 'haploid':
        df[new_col] = [random.random() < (base_freq*rel if is_case else base_freq) for is_case in df[case_col]]
    else:
        df[new_col] = [int(random.random() < (base_freq*rel if is_case else base_freq)) + int(random.random() < (base_freq*rel if is_case else base_freq)) for is_case in df[case_col]]
        if locus_model == 'dominant':
            df[new_col].replace({2:1, 1:1, 0:0}, inplace=True)
        elif locus_model == 'recessive':
            df[new_col].replace({2:1, 1:0, 0:0}, inplace=True)

def case_control_split(df, new_col, fraction, n):
    df[new_col] = [random.random() < fraction for i in range(n)]

def shuffle(df, colname, select_rows=None):
    if select_rows is None:
        values = list(df[colname])
        random.shuffle(values)
        df[colname] = values
    else:
        values = list(df.loc[select_rows, colname])
        random.shuffle(values)
        df.loc[select_rows, colname] = values

def OR(df, colname, maxincol):
    dfS = df[df['outcome']==True]
    dfH = df[df['outcome']==False]
    if colname in ('A', 'B'):
        return dfS[colname].sum() / dfH[colname].sum() / abs(maxincol-dfS[colname]).sum() * abs(maxincol-dfH[colname]).sum()
    else:
        return dfS[colname].sum() / dfH[colname].sum() / dfS['A0B0'].sum() * dfH['A0B0'].sum()

def RR(df, colname, maxincol):
    dfS = df[df['outcome']==True]
    dfH = df[df['outcome']==False]
    if colname in ('A', 'B'):
        return dfS[colname].sum() / df[colname].sum() / abs(maxincol-dfS[colname]).sum() * abs(maxincol-df[colname]).sum()
    else:
        return dfS[colname].sum() / df[colname].sum() / dfS['A0B0'].sum() * df['A0B0'].sum()

def sample_ctrls(ctrls_to_cases, df):
    dfS = df[df['outcome']==True]
    dfH = df[df['outcome']==False]
    if len(dfS)*ctrls_to_cases < len(dfH):
        dfH = dfH.sample(n=round(len(dfS)*ctrls_to_cases))
    elif len(dfS)*ctrls_to_cases > len(dfH):
        dfS = dfS.sample(n=round(len(dfH)/ctrls_to_cases))
    return pandas.concat([dfH, dfS], ignore_index=True)

def calc_observed(df, locus_modelA, locus_modelB):
    maxA = 2 if locus_modelA == 'additive_codominant' else 1
    maxB = 2 if locus_modelB == 'additive_codominant' else 1
    maxAB = maxA * maxB
    
    df['A'] = df['A'].astype(int)
    df['B'] = df['B'].astype(int)
    df['A1B1'] = df['A'] * df['B']
    df['A1B0'] = df['A'] * (maxB-df['B'])
    df['A0B1'] = (maxA-df['A']) * df['B']
    df['A0B0'] = (maxA-df['A']) * (maxB-df['B'])
    
    OR11 = OR(df, 'A1B1', maxAB)
    OR10 = OR(df, 'A1B0', maxAB)
    OR01 = OR(df, 'A0B1', maxAB)
    RR11 = RR(df, 'A1B1', maxAB)
    RR10 = RR(df, 'A1B0', maxAB)
    RR01 = RR(df, 'A0B1', maxAB)
    
    return OR11, OR10, OR01, RR11, RR10, RR01

if '__main__' == __name__:
    model_mapping = {'Synergism':AND_logic_sim_modelIV, 'Heterogeneity':OR_logic_sim_modelV, 'ConfI':same_cause_sim_modelI, 'ConfII':two_group_with_different_prevalence,  'AddR':separate2_cause_sim_modelII, 'AddO':separate_cause_sim_modelIII, 'Mthreshold_t1of5':functools.partial(threshold_three_of_five, threshold=1), 'Mthreshold_t2of5':functools.partial(threshold_three_of_five, threshold=2), 'Mthreshold_t3of5':functools.partial(threshold_three_of_five, threshold=3), 'Mthreshold_t4of5':functools.partial(threshold_three_of_five, threshold=4), 'Mthreshold_t5of5':functools.partial(threshold_three_of_five, threshold=5)}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=model_mapping.keys())
    parser.add_argument('prevalence', type=float)
    parser.add_argument('freqX', type=float)
    parser.add_argument('freqXbg', type=float)
    parser.add_argument('freqY', type=float)
    parser.add_argument('freqYbg', type=float)
    parser.add_argument('-n', '--n_simulated_individuals', type=int, default=1000000, help='default: 1 million')
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--biased_sampling', nargs='?', const=1, type=float, metavar='ctrls_to_cases')
    parser.add_argument('-arX', '--allele_relationship_X', choices=['haploid', 'dominant', 'recessive', 'additive_codominant'], default='haploid', help='default: haploid i.e. binary set directly by the frequencies')
    parser.add_argument('-arY', '--allele_relationship_Y', choices=['haploid', 'dominant', 'recessive', 'additive_codominant'], default='haploid', help='default: haploid i.e. binary set directly by the frequencies')
    o = parser.parse_args()
    
    if o.random_seed: random.seed(o.random_seed)
    
    simdata = model_mapping[o.model](o.freqX/o.freqXbg, o.freqY/o.freqYbg, o.n_simulated_individuals, o.freqXbg, o.freqYbg, o.prevalence, o.allele_relationship_X, o.allele_relationship_Y)
    
    if o.biased_sampling is not None:
        simdata = sample_ctrls(o.biased_sampling, simdata)
    
    OR11, OR10, OR01, RR11, RR10, RR01 = calc_observed(simdata, o.allele_relationship_X, o.allele_relationship_Y)
    
    print('parameters:', o)
    
    print('RR11:', RR11)
    print('RR10:', RR10)
    print('RR01:', RR01)
    print('RR01 * RR10:', RR01*RR10)
    print('RR01 + RR10 - 1:', RR01 + RR10 - 1)
    
    print('OR11:', OR11)
    print('OR10:', OR10)
    print('OR01:', OR01)
    print('OR01 * OR10:', OR01*OR10)
    print('OR01 + OR10 - 1:', OR01 + OR10 - 1)
    
    print('√(f)estRR:', (RR11 - RR10 - RR01 + 1)/(RR10-1)/(RR01-1))
    print('√(f)est:', (OR11 - OR10 - OR01 + 1)/(OR10-1)/(OR01-1))
    print('RERI:', OR11 - OR10 - OR01 + 1)
    print('AP:', (OR11 - OR10 - OR01 + 1)/OR11)
    print('Synergy index:', (OR11 - 1)/(OR10-1 + OR01-1))