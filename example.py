# Example of running PyQSpecFit from special parameter file
import PyQSpecFit
import pandas as pd
import matplotlib.pyplot as plt

file = 'Run_Files/runFile.csv'
pdata = pd.read_csv(file)

example = PyQSpecFit.PyQSpecFit()

# Perform fits #
example.runFile(file)
example.evalFile(file)
example.plotFile(file)





