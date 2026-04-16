Neon
====

Execution Throughput
--------------------

Microbenchmark the execution throughput of the following instructions:

* ``FMADD (scalar)``, FP32 variant.
* Optional:
   * ``FMLA (vector)`` with arrangement specifier ``4S``.
   * ``FMLA (vector)`` with arrangement specifier ``2S``.
 

Permutation
-----------

This section develops a kernel that performs a permutation operation on the dimensions of a tensor :math:`abc` to obtain a tensor :math:`cba`: :math:`abc \rightarrow cba`.
The two tensors are stored in row-major order. Dimension :math:`a` has size :math:`|a|=8` and :math:`b` has size :math:`|b|=4`. The size of dimension :math:`c` is a function parameter.  

The kernel has the following signature:

.. code-block:: C

  /**
   * @brief Permutation operation abc->cba
   * @param size_c Size of dimension c.
   * @param abc    Pointer to row-major tensor abc.
   * @param cba    Pointer to row-major tensor cba.
   **/
  void perm_neon_abc_cba(int64_t       size_c
                         float const * abc,
                         float       * cba);

.. admonition:: Tasks

   1. Implement a Neon kernel that performs :math:`abc \rightarrow cba`.
   2. Optimize your kernel. Report its performance in GiB/s in dependency of size :math:`|c|` .