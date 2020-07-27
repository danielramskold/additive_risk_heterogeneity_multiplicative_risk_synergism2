import matplotlib
matplotlib.use('Agg')
import pandas, random, math, argparse, numpy
from collections import defaultdict, OrderedDict
from matplotlib import pyplot
from functools import partial

def spike_in(df, case_col, new_col, rel, base_freq):
    df[new_col] = [random.random() < (base_freq*rel if is_case else base_freq) for is_case in df[case_col]]

def case_control_split(df, new_col, fraction, n):
    df[new_col] = [random.random() < fraction for i in range(n)]

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

def shuffle(df, colname, select_rows=None):
    if select_rows is None:
        values = list(df[colname])
        random.shuffle(values)
        df[colname] = values
    else:
        values = list(df.loc[select_rows, colname])
        random.shuffle(values)
        df.loc[select_rows, colname] = values

def invert(v):
    return 1/v

def calc_observed_expected(df, useRR):
    maxA = 1
    maxB = 1
    maxAB = maxA * maxB
    
    df['A'] = df['A'].astype(int)
    df['B'] = df['B'].astype(int)
    df['A1B1'] = df['A'] * df['B']
    df['A1B0'] = df['A'] * (maxB-df['B'])
    df['A0B1'] = (maxA-df['A']) * df['B']
    df['A0B0'] = (maxA-df['A']) * (maxB-df['B'])
    
    ORorRR = RR if useRR else OR
    expected_OR11_additive = ORorRR(df, 'A1B0', maxAB) + ORorRR(df, 'A0B1', maxAB) - 1
    expected_OR11_multiplicative = ORorRR(df, 'A1B0', maxAB) * ORorRR(df, 'A0B1', maxAB)
    observed_OR11 = ORorRR(df, 'A1B1', maxAB)
    if o.correlation:
        from scipy import stats
        dfS = df[df['outcome']==True]
        dfH = df[df['outcome']==False]
        try:
            return stats.pearsonr(df['A'], df['B'])[0], stats.pearsonr(dfS['A'], dfS['B'])[0], stats.pearsonr(dfH['A'], dfH['B'])[0]
        except ValueError:
            return None, None, None
    else:
        return (observed_OR11/expected_OR11_multiplicative), (observed_OR11/expected_OR11_additive)


def separate_cause_sim(a_rel, b_rel, n, bg_freq1, bgfreq2, prevalence):
    df_a = pandas.DataFrame()
    df_b = pandas.DataFrame()
    case_control_split(df_a, 'outcome', prevalence, n//2)
    case_control_split(df_b, 'outcome', prevalence, n-n//2)
    spike_in(df_a, 'outcome', 'A', a_rel, bg_freq1)
    spike_in(df_b, 'outcome', 'A', 1, bg_freq1)
    spike_in(df_a, 'outcome', 'B', 1, bg_freq2)
    spike_in(df_b, 'outcome', 'B', b_rel, bg_freq2)
    df = pandas.concat((df_a, df_b), ignore_index=True)
    return df

def separate2_cause_sim(a_rel, b_rel, n, bg_freq1, bgfreq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'outcome', prevalence, n)
    spike_in(df, 'outcome', 'A', a_rel, bg_freq1)
    spike_in(df, 'outcome', 'B', b_rel, bg_freq2)
    case_control_split(df, 'group', 0.5, n)
    shuffle(df, 'A', df['group'])
    shuffle(df, 'B', ~df['group'])
    return df

def separate2_cause_sim_v2(a_rel, b_rel, n, bg_freq1, bgfreq2, prevalence):
    df_a = pandas.DataFrame()
    df_b = pandas.DataFrame()
    case_control_split(df_a, 'outcome', prevalence, n//2)
    case_control_split(df_b, 'outcome', prevalence, n-n//2)
    spike_in(df_a, 'outcome', 'A', a_rel, bg_freq1)
    spike_in(df_b, 'outcome', 'A', a_rel, bg_freq1)
    spike_in(df_a, 'outcome', 'B', b_rel, bg_freq2)
    spike_in(df_b, 'outcome', 'B', b_rel, bg_freq2)
    shuffle(df_a, 'B')
    shuffle(df_b, 'A')
    df = pandas.concat((df_a, df_b), ignore_index=True)
    return df

def same_cause_sim(a_rel, b_rel, n, bg_freq1, bgfreq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'outcome', prevalence, n)
    spike_in(df, 'outcome', 'A', a_rel, bg_freq1)
    spike_in(df, 'outcome', 'B', b_rel, bg_freq2)
    return df

def OR_logic_sim(a_rel, b_rel, n, bg_freq1, bgfreq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'a_cases', 1-math.sqrt(1-prevalence), n)
    case_control_split(df, 'b_cases', 1-math.sqrt(1-prevalence), n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2)
    df['outcome'] = df['a_cases'] | df['b_cases']
    return df

def AND_logic_sim(a_rel, b_rel, n, bg_freq1, bgfreq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'a_cases', math.sqrt(prevalence), n)
    case_control_split(df, 'b_cases', math.sqrt(prevalence), n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2)
    df['outcome'] = df['a_cases'] & df['b_cases']
    return df

def threshold_three_of_five(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence, threshold):
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
    
    
    case_control_split(df, 'a_cases', subcasefreq, n)
    case_control_split(df, 'b_cases', subcasefreq, n)
    case_control_split(df, 'c_cases', subcasefreq, n) # for prevalence=0.5, **3 for threshold 1, **1.71 for threshold 2, **1 for threshold 3, **0.53 fo threshold 4, **0.2 for threshold 5
    case_control_split(df, 'd_cases', subcasefreq, n)
    case_control_split(df, 'e_cases', subcasefreq, n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2)
    df['outcome'] = df['a_cases'].astype(int) + df['b_cases'].astype(int) + df['c_cases'].astype(int) + df['d_cases'].astype(int) + df['e_cases'].astype(int) >= threshold
    return df

def add_to_dict(d, key_prefix, values):
    if None in values: return
    if o.correlation:
        keys = [key_prefix+'\n'+suffix for suffix in ('rall', 'rcase', 'rctrl')]
    else:
        keys = [key_prefix+'\n'+suffix for suffix in ('mult', 'add', 'het')]
    for k, v in zip(keys, values):
        if k not in d: d[k] = []
        d[k].append(v)

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('calc', choices=['RR', 'OR'])
parser.add_argument('protective', choices=['no', 'one', 'two'])
parser.add_argument('-n', default=[1000, 1000000], type=int, nargs=2)
parser.add_argument('--correlation', action='store_true')
parser.add_argument('--vocal', action='store_true')
parser.add_argument('--notch', action='store_true')
o = parser.parse_args()

nr, n= o.n
if o.protective == 'two':
    bg_freq_arr1 = [0.05+random.random()*0.4 for i in range(nr)]
    bg_freq_arr2 = [0.05+random.random()*0.2 for i in range(nr)]
    a_rel_arr = [1/(1.1+random.random()*3.9) for i in range(nr)]
    b_rel_arr = [1/(1.1+random.random()*0.9) for i in range(nr)]
elif o.protective == 'one':
    bg_freq_arr1 = [0.05+random.random()*0.1 for i in range(nr)]
    bg_freq_arr2 = [0.05+random.random()*0.2 for i in range(nr)]
    a_rel_arr = [1.1+random.random()*3.9 for i in range(nr)]
    b_rel_arr = [1/(1.1+random.random()*0.9) for i in range(nr)]
elif o.protective == 'no':
    bg_freq_arr1 = [0.05+random.random()*0.1 for i in range(nr)]
    bg_freq_arr2 = [0.05+random.random()*0.2 for i in range(nr)]
    a_rel_arr = [1.1+random.random()*3.9 for i in range(nr)]
    b_rel_arr = [1.1+random.random()*0.9 for i in range(nr)]

prev_obs = defaultdict(list)
prev_obs_deviation = OrderedDict()
log2_obs_to_exp = OrderedDict()
for prevalence in (0.5, 0.15, 0.05, 0.015, 0.005):
    for a_rel, b_rel, bg_freq1, bg_freq2 in zip(a_rel_arr, b_rel_arr, bg_freq_arr1, bg_freq_arr2):
        add_to_dict(log2_obs_to_exp, '%.3f\nsynth'%prevalence, calc_observed_expected(AND_logic_sim(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence), o.calc=='RR'))
        add_to_dict(log2_obs_to_exp, '%.3f\nhetero'%prevalence, calc_observed_expected(OR_logic_sim(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence), o.calc=='RR'))
        for threshold in [1,2,3,4,5]:
            df = threshold_three_of_five(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence, threshold)
            prev_obs[ '%.3f\nt=%i'%(prevalence, threshold)].append(numpy.mean(df['outcome']))
            prev_obs_deviation[abs(numpy.mean(df['outcome'])-prevalence)/prevalence] = '%.3f\nt=%i'%(prevalence, threshold)
            add_to_dict(log2_obs_to_exp, '%.3f\nt=%i'%(prevalence, threshold), calc_observed_expected(df, o.calc=='RR'))

print({k:numpy.median(V) for k,V in prev_obs.items()}, max(prev_obs_deviation.keys()), repr(prev_obs_deviation[max(prev_obs_deviation.keys())]))

print({k:v for k,v in prev_obs_deviation.items() if k>0.1})

xarr = list(range(1,1+len(log2_obs_to_exp)))
if o.correlation:
    pyplot.gca().axhline(0)
else:
    pyplot.gca().axhline(1)
pyplot.boxplot(list(log2_obs_to_exp.values()), sym='', notch=o.notch)
pyplot.xticks(xarr, list(log2_obs_to_exp.keys()), fontsize=2)
if o.correlation:
    pyplot.ylabel('correlation between risk factors')
else:   
    pyplot.ylabel('observed/expected for %s(A1B1)'%o.calc)
    pyplot.ylim(0, 2.5)
#pyplot.xlim(min(xarr)-1, min(xarr)+16)
pyplot.savefig('plot%s_for_causal_spike_in_models_%s_%sprotective%s.pdf'%('_notched' if o.notch else '', 'correlation' if o.correlation else o.calc, o.protective, '_n=%i*%i'%tuple(o.n) if o.n != [1000, 1000000] else ''))