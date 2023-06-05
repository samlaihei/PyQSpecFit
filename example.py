# Example of running PyQSpecFit from special parameter file
import PyQSpecFit
import timeit

start = timeit.default_timer()


file = 'Run_Files/runFile.csv'

example = PyQSpecFit.PyQSpecFit()

# Perform fits #
example.runFile(file)
example.evalFile(file)
example.plotFile(file)


print("Time taken is :",
              timeit.default_timer() - start)


