# Notes on learning CuTeDSL

- Need a flag for PTX: `CUTE_DSL_LINEINFO=1 ncu --set full -o cute_soft uv run softmax.py`
- Very easy to break the machine 
- Docs are dogshit 
- Tiers of memory hierarchy exposed
- fragment == tensor in registers
- TV layout is very powerful
- LDG128 is BITS not bytes

## Beating Triton with CuTeDSL

Triton actually seems to codegen really good code. On my 3090 we can see that my CuteDSL softmax is 2us slower! 
