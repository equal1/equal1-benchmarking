# Equal1 Benchmarking

Benchmarking scripts to verify the performance of an Equal1 Quantum Computer

QPU technical features

Bypass transpilation 
    -> RxP(duration, frequency, amplitude) -> X pulse duration 
        -> freq and amplitude can burn the device, dont allow access. 
    -> RxP(angle) -> X pulse duration
Optimization level 0
    -> Rx(theta) -> Rx(pi/2) + Rz(theta) + Rx(-pi/2)


[ ] Qubit individual addressability
    -> Rabi oscillations need Delay gate? 
[x] T1, T2*, T2 Hahn Echo, T2 CPMG
    -> Delay Gate 
[x] 1Q, 2Q randomised benchmarking
[x] SPAM and measurement fidelity
[ ] T2* in the shuttling zone
    -> Shuttle Gate
[x] Bell state fidelity and concurrence
[?] Bell state with shuttling  fidelity and concurrence
    -> Shuttle Gate
[x] Gate set tomography
[X] Connectivity and direct 2Q connection benchmarking (additional benchmark)

Applications

[o]  GHZ state
[x]  QAOA for MAX-CUT 
[o]  Bernstein-Vazerani Problem
[x]  3 GHZ states with shuttling
[ ]  CLOPS (additional benchmark)
    -> Clops H is the new one from IBM 

External Benchmarks

[x]  Application-oriented volumetric benchmarking (additional benchmark)


Note: for the application-oriented volumetric benchmarking please make sure you also init git submodules:
```bash
git submodule update --init --recursive
```


Status of Delay Gate

What does a shuttling gate look like: 
* is it Shuttle(qubit, target_position) 
* is it Shuttle(qubit, distance)