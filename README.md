Simulations python3 code:
* spike_in_plots_causal_with_biased_sampling.py simulates a case-control setup on top of a logical AND, OR or threshold model
* spike_in_plots_causal.py simulates a logical AND, OR or threshold model
* spike_in_plot_3loci.py is an example of three risk factors (Fig 2B-C)
* spike_in_plots_confounding.py simulates to confounder scenarios
* spike_in_plots_additive.py attempt and succeeds at generating precicely additive odds ratios or relative risks


Rheumatoid arthritis python3 code:
* calc_risk_square_modelrandomizations.py is code for generating odds ratios, including for two permutation controls
* fewsnps_obstoexp_distribution.ipynb is a Jupyter notebook for matching known risk SNPs to additive or multiplicative effect, based on the output from calc_risk_square_modelrandomizations.py
* eira_randomised_distibutions.ipynb plots all SNPs from the output of calc_risk_square_modelrandomizations.py
* SE.ipynb plots the output from GEISA
* correlationtriplets.py is similar to calc_risk_square_modelrandomizations.py but calculates correlations between risk factors
* correlations.ipynb plots the results from correlationtriplets.py


Other:
* Try_Pairwise_Scenarios.py does not produce plots, but instead allows the user to play around with numbers to try out the models of spike_in_plots_causal_with_biased_sampling.py, spike_in_plots_causal.py, spike_in_plots_confounding.py and spike_in_plots_additive.py.
* dr_tools.py is support code (library) for the other ones

All programs run in Python 3.7, and some have been tested in Python 3.5.
