import numpy as np
import GlobalOptimizationHRLA as GO

samples_filename10 = "temp_output/data/HRLA10_1733409408.919645.pickle"
samples_filename20 = "temp_output/data/HRLA20_1733413013.7516716.pickle"

comparator = GO.Comparator([samples_filename10, samples_filename20])
comparator.plot_empirical_probabilities_per_d(dpi=100, tols=[2,4], running=False)

postprocessor10 = GO.PostProcessor(samples_filename10)
postprocessor10.plot_empirical_probabilities(dpi=100, layout="32", tols=[3,4,5,6,7,8], running=False)

postprocessor20 = GO.PostProcessor(samples_filename20)
postprocessor20.plot_empirical_probabilities(dpi=100, layout="32", tols=[3,4,5,6,7,8], running=False)
