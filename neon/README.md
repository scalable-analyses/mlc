Neon
====

Execution Throughput
--------------------

Microbenchmark the execution throughput of the following instructions:

* `FMADD (scalar)`, FP32 variant.
* Optional:
   * `FMLA (vector)` with arrangement specifier `4S`.
   * `FMLA (vector)` with arrangement specifier `2S`.
 

Permutation
-----------

This section develops a kernel that performs a permutation operation on the dimensions of a tensor $abc$ to obtain a tensor $cba$: $abc \rightarrow cba$.
The two tensors are stored in row-major order. Dimension $a$ has size $|a|=8$ and $b$ has size $|b|=4$. The size of dimension $c$ is a function parameter.  

The kernel has the following signature:

```c

  /**
   * @brief Permutation operation abc->cba
   * @param size_c Size of dimension c.
   * @param abc    Pointer to row-major tensor abc.
   * @param cba    Pointer to row-major tensor cba.
   **/
  void perm_neon_abc_cba(int64_t       size_c
                         float const * abc,
                         float       * cba);
```
### Tasks

   1. Implement a Neon kernel that performs $abc \rightarrow cba$.
   2. Optimize your kernel. Report its performance in GiB/s in dependency of size $|c|$ .