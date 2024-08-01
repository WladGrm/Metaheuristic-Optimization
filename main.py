import opt

"""
alg = opt.ACO()
alg.read_parameters_file("parameters.json")
print(alg)
alg.solve()
alg.plot_path()
"""
sa_alg = opt.SA()
sa_alg.read_parameters_file("parameters.json")
sa_alg.solve()
sa_alg.plot()
