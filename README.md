# Equal1 Benchmarking

Benchmarking scripts to verify the performance of an Equal1 Quantum Computer

## Devices

- [Bell 1](docs/devices/bell1-6qubit.ipynb)
- [Bell 2](docs/devices/bell2.ipynb)

## QPU technical features

### Hardware only tests

- [ ] Qubit individual addressability (Rabi oscillations)

### Qubit Quality Benchmarks

- [x] [T1 Time](docs/benchmarks/t1_time.ipynb)
- [x] [T2* Time](docs/benchmarks/t2_time.ipynb)
- [ ] T2 Hahn Echo, T2 CPMG
- [ ] 1Q, 2Q randomised benchmarking
- [ ] SPAM and measurement fidelity
- [ ] Gate set tomography
- [ ] Connectivity and direct 2Q connection benchmarking 

### Applications

- [x]  [GHZ state](docs/benchmarks/ghz_states.ipynb)
- [x]  [QAOA for MAX-CUT (QScore)](docs/benchmarks/qaoa-maxcut.ipynb)
- [x]  [Bernstein-Vazerani Problem](docs/benchmarks/bernstein-vazirani.ipynb)

### External Benchmarks

- [ ]  CLOPS (additional benchmark)
- [x]  [Application-oriented volumetric benchmarking](docs/QC-app-oriented-benchmark/benchmark_hpc_gamma1.ipynb) 

### Getting started

Note: for the application-oriented volumetric benchmarking please make sure you also init git submodules:
```bash
git submodule update --init --recursive
```
