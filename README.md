# Equal1 Benchmarking

Benchmarking scripts to verify the performance of an Equal1 Quantum Computer

QPU technical features

Bypass transpilation 
    -> RxP(duration, frequency, amplitude) -> X pulse duration 
        -> freq and amplitude can burn the device, dont allow access. 
    -> RxP(angle) -> X pulse duration
Optimization level 0
    -> Rx(theta) -> Rx(pi/2) + Rz(theta) + Rx(-pi/2)

Hardware only test 
[ ] Qubit individual addressability
    -> Rabi oscillations need RxP gate?

Qubit Quality Benchmarks 
[A] T1, T2*,
 T2 Hahn Echo, T2 CPMG
    -> Delay Gate 
[?] T2 Hahn Echo, T2 CPMG
[A] 1Q, 2Q randomised benchmarking
[A] SPAM and measurement fidelity
[-] T2* in the shuttling zone
    -> Shuttle Gate
[N] Bell state fidelity and concurrence
[?] Bell state with shuttling  fidelity and concurrence
    -> Shuttle Gate
[-] Gate set tomography
[?] Connectivity and direct 2Q connection benchmarking (additional benchmark)

Applications
[o]  GHZ state
[N]  QAOA for MAX-CUT 
[o]  Bernstein-Vazerani Problem
[-]  3 GHZ states with shuttling

External Benchmarks
[-]  CLOPS (additional benchmark)
    -> Clops H is the new one from IBM 
[-]  Application-oriented volumetric benchmarking (additional benchmark)


Status of Delay Gate

What does a shuttling gate look like: 