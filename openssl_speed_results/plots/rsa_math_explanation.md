# Mathematical Operations in OpenSSL RSA Benchmarks

## RSA-2048 Operations

The OpenSSL 'speed' benchmark for RSA-2048 performs four main cryptographic operations:

1. **Sign**: Uses private key to create digital signatures
   - Mathematical representation: m^d mod n
   - Private key operation (slower)
   - Estimated arithmetic operations: ~3.16 million per RSA-2048 sign

2. **Verify**: Uses public key to verify signatures
   - Mathematical representation: s^e mod n
   - Public key operation (faster due to small public exponent)
   - Estimated arithmetic operations: ~170,000 per RSA-2048 verify

3. **Encrypt**: Uses public key to encrypt data
   - Mathematical representation: m^e mod n
   - Public key operation (faster)
   - Estimated arithmetic operations: ~170,000 per RSA-2048 encrypt

4. **Decrypt**: Uses private key to decrypt data
   - Mathematical representation: c^d mod n
   - Private key operation (slower)
   - Estimated arithmetic operations: ~3.16 million per RSA-2048 decrypt

## Core Mathematical Operations

The fundamental mathematical operation in RSA is **modular exponentiation** (calculating a^b mod n).
This involves:

- Modular multiplications
- Modular squaring operations
- Modular reductions

For a 2048-bit RSA key:
- The modulus n is 2048 bits long
- A full exponentiation requires approximately 3,072 modular multiplications for private key operations
- Each modular multiplication requires approximately 1,000 elementary operations
- Public key operations are much faster because the public exponent is typically small (usually 65537)

## OpenSSL Optimizations

OpenSSL implements several optimizations for these operations:

1. **Chinese Remainder Theorem (CRT)**: Speeds up private key operations by about 4x
2. **Montgomery multiplication**: Reduces the cost of modular reductions
3. **Windowing techniques**: Reduce the number of multiplications needed in exponentiation
4. **Assembly optimizations**: Hardware-specific code paths for maximum performance

## Total Computational Work

A typical 20-second OpenSSL RSA-2048 test on a single thread performs:
- ~213 sign operations
- ~213 decrypt operations
- ~4,096 verify operations
- ~2,048 encrypt operations

This translates to approximately 1.4 billion elementary arithmetic operations per second on a modern CPU core.
When using multiple threads, this computational work scales nearly linearly until hardware limits are reached.
