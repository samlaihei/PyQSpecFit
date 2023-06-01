# Example of running PyQSpecFit from special parameter file
import PyQSpecFit

file = 'Run_Files/runFile.csv'

example = PyQSpecFit.PyQSpecFit()

# Perform fits #
example.runFile(file)
example.evalFile(file)
example.plotFile(file)





