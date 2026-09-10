# The Fidelius Protocol
 **A Lattice-Based Confidential UTXO Architecture with Asynchronous Addressing and State Cut-Through**

Fidelius is a post-quantum evolution of the Mimblewimble transaction topology (commitments, excesses, kernels, cut-through) built entirely over Module-SIS. 

This repository contains a complete, self-validating, executable Python specification of the protocol. **Nothing is stubbed.** From the ring arithmetic and exact balance arguments to a full implementation of ML-KEM-768 (FIPS 203), the entire cryptographic suite is tested and actively subjected to structural attack simulations upon execution.

## Key Cryptographic Innovations

1. **Exact Balance Extraction (The $\Phi$-Homomorphism):** 
   Lattice Fiat-Shamir proofs typically yield a "relaxed" extraction, leaving a gap in proving a zero-sum transaction balance. This protocol closes that gap. By applying the ring homomorphism $\Phi(u) = u(2) \pmod{2^{256}+1}$, the relaxed identity is translated into exact integer arithmetic, reducing the probability of forging balance to $2^{-206}$.
   
2. **Asynchronous Offline Payments:**
   Traditional Mimblewimble requires both sender and receiver to interact to negotiate blinding factors, making batching impossible. Fidelius decouples the blinding factor from spend authority. Using ML-KEM-768, the sender securely encapsulates the blinding factor offline, while the receiver maintains a distinct ownership key bound to the specific input/transaction. 
   
3. **Arithmetic Verifier (SNARK/STARK Ready):**
   The Fiat-Shamir heuristic utilizes a Rescue-style algebraic sponge over $R_q$ (rather than SHAKE). Because the entire verifier evaluates as $R_q$ arithmetic, it can be efficiently expressed inside recursive proof systems.

4. **Honest LNP22 Machinery Benchmarks:**
   The specification implements working sub-components of LNP22 (constant-coefficient proofs, ring challenges, exact norm proofs). It formally proves that at the $q \approx 2^{57}$ modulus required by the $\Phi$-balance argument, a faithful LNP22 range proof requires ~40 KB, updating theoretical projections.

## How to Run the Executable Spec

The entire protocol, including cryptographic parameters, security estimators, block application, and a rigorous attack test suite, lives in a single file.

### Prerequisites
* Python 3.x
* `numpy`

### Execution
Run the full test suite and attack simulations:
```bash
python3 demonstration.py
```
*(For a faster run with fewer repetitions, use `python3 demonstration.py --quick`)*

Exit code is `0` if and only if every mathematical claim, estimator, and consensus check passes.

## What is Verified Internally?
Upon execution, the script automatically tests:
- **T01-T03**: All classical core-SVP hardness estimators (Module-SIS / MLWE), $R_q$ exact integer arithmetic, and statistical hiding bounds.
- **T04**: The $\Phi$ Algebra, carry-certificates, and the relaxed-extraction balance arguments.
- **T05-T07**: Full FIPS-203 ML-KEM validation, Non-interactive payments, and Range/Kernel proofs.
- **T08**: Ledger operations, state cut-through, and kernel offsets preventing un-aggregation.
- **T09 (Attacks)**: A massive structural attack suite validating that unbalanced values, out-of-bounds norms, duplicate UTXOs, and range-proof grinding fail consensus.
- **T11-T13**: PRF-derived Memory-hard PoW, LNP22 components, and Exact Norm Proofs.

## Future Work & Disclaimers

This is a mathematical proof-of-concept and cryptographic specification, **not a production node**. 
* **Sponge Cryptanalysis:** The algebraic Rescue-style sponge lacks the symmetric cryptanalysis required for production use (recommend transitioning to Poseidon2/Rescue-Prime pending analysis).
* **Constant Time:** The Python implementation prioritizes clarity and mathematical exactness; it makes no attempt to execute in constant time.
* **Soundness Trapdoors:** The multi-party DOTT protocol utilizes plain hiding commitments. Provable formal zero-knowledge would require an upgrade to trapdoor commitments. 
