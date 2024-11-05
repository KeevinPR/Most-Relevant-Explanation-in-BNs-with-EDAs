# %%
# Imports
from pgmpy.readwrite import BIFReader
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from itertools import product
from EDAspy.optimization import UMDAcat
import random
import warnings
import logging
import time
import json
import mre

# %%
# Import the Bayesian Network as a BIF file
reader = BIFReader('/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/MREWithEDAs/asia.bif')  # Ensure 'asia.bif' is in the same directory
model = reader.get_model()

# %%
# Generating a example of target and evidence
leaves = model.get_leaves()
rest =  [x for x in model.states.keys()  if x not in leaves]
leaves.remove('xray')
rest.append('xray')
print(leaves)
print(rest)
warnings.filterwarnings("ignore")
logger = logging.getLogger("pgmpy")
logger.setLevel(logging.ERROR)
target_names = random.sample(rest,7)
sim = model.simulate(1,show_progress=False)
evidence_values = sim[leaves].values.tolist()[0]
evidence = {n:v for n,v in zip(leaves,evidence_values)}
d = {'target':target_names,'evidence':evidence}

# %%
# Example generated
print(d)

# %%
s = time.time()
sol,gbf,_ = mre.UMDAcat_mre2(model,d['evidence'],d['target'],size_gen=50,dead_iter=20,verbose=False,alpha=0.8,best_init=True)
e = time.time()
print({'sol':sol,'gbf':gbf,'time':e-s})

# %%
s = time.time()
sol,gbf = mre.dea_mre(model,d['evidence'],d['target'],50,5000)
e = time.time()
print({'sol':sol,'gbf':gbf,'time':e-s})

# %%
s = time.time()
sol,gbf,_ = mre.ebna_mre(model,d['evidence'],d['target'],size_gen=50,dead_iter=20,verbose=False,alpha=0.8,best_init=True)
e = time.time()
print({'sol':sol,'gbf':gbf,'time':e-s})

# %%
s = time.time()
sol,gbf = mre.es_mre(model,d['evidence'],d['target'],50,5000)
e = time.time()
print({'sol':sol,'gbf':gbf,'time':e-s})

# %%
s = time.time()
sol,gbf = mre.ga_mre(model,d['evidence'],d['target'],50,5000)
e = time.time()
print({'sol':sol,'gbf':gbf,'time':e-s})

# %%
s = time.time()
sol,gbf = mre.hierarchical_beam_search(model,d['evidence'],d['target'],5,1+1e-08,10,2)
e = time.time()
print({'sol':sol,'gbf':gbf,'time':e-s})

# %%
s = time.time()
sol,gbf = mre.nsga2_mre(model,d['evidence'],d['target'],pop_size=50,n_gen=50,best_init=True,period=10)
e = time.time()
print({'sol':sol,'gbf':gbf,'time':e-s})

# %%
s = time.time()
sol,gbf = mre.pso_mre(model,d['evidence'],d['target'],50,5000)
e = time.time()
print({'sol':sol,'gbf':gbf,'time':e-s})

# %%
s = time.time()
sol,gbf = mre.tabu_mre(model,d['evidence'],d['target'],200,30,more_targets=1)
e = time.time()
print({'sol':sol,'gbf':gbf,'time':e-s})


