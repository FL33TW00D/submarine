# Submarine

# profile a CUDA kernel
# remember to compile with -lineinfo
ncu --set full --import-source on -o profile python main.py

# debug illegal memory access
compute-sanitizer python main.py
