# Equal1 Benchmarking

Benchmarking scripts to verify the performance of an Equal1 Quantum Computer

## QPU technical features

### Hardware only tests

- [ ] Qubit individual addressability (Rabi oscillations)

### Qubit Quality Benchmarks

- [x] T1, T2*,
- [ ] T2 Hahn Echo, T2 CPMG
- [A] 1Q, 2Q randomised benchmarking
- [ ] SPAM and measurement fidelity
- [ ] T2* in the shuttling zone
- [ ] Bell state fidelity and concurrence
- [ ] Bell state with shuttling  fidelity and concurrence
- [ ] Gate set tomography
- [ ] Connectivity and direct 2Q connection benchmarking (additional benchmark)

### Applications

- [x]  GHZ state
- [ ]  QAOA for MAX-CUT (QScore)
- [x]  Bernstein-Vazerani Problem
- [ ]  3 GHZ states with shuttling

### External Benchmarks

- [ ]  CLOPS (additional benchmark)
- [x]  Application-oriented volumetric benchmarking (additional benchmark)

### Getting started

Note: for the application-oriented volumetric benchmarking please make sure you also init git submodules:
```bash
git submodule update --init --recursive
```
