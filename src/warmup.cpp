//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-758885
//
// All rights reserved.
//
// This file is part of Comb.
//
// For details, see https://github.com/LLNL/Comb
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////

#include "comb.hpp"

namespace COMB {

template < typename pol >
void do_warmup(ExecContext<pol> const& con, COMB::Allocator& aloc, Timer& tm, IdxT num_vars, IdxT len)
{
  tm.clear();

  char test_name[1024] = ""; snprintf(test_name, 1024, "warmup %s %s", pol::get_name(), aloc.name());
  Range r(test_name, Range::green);

  DataT** vars = new DataT*[num_vars];

  for (IdxT i = 0; i < num_vars; ++i) {
    vars[i] = (DataT*)aloc.allocate(len*sizeof(DataT));
  }

  for (IdxT i = 0; i < num_vars; ++i) {

    DataT* data = vars[i];

    for_all(con, 0, len, detail::set_n1{data});
  }

  synchronize(con);
}

void warmup(COMB::ExecContexts& exec,
            COMB::Allocators& alloc,
            Timer& tm, IdxT num_vars, IdxT len)
{
  // warm-up memory pools
  do_warmup(exec.seq, alloc.host, tm, num_vars, len);

#ifdef COMB_ENABLE_OPENMP
  do_warmup(exec.omp, alloc.host, tm, num_vars, len);
#endif

#ifdef COMB_ENABLE_CUDA
  do_warmup(exec.seq, alloc.hostpinned, tm, num_vars, len);

  do_warmup(exec.cuda, alloc.device, tm, num_vars, len);

  do_warmup(exec.seq,  alloc.managed, tm, num_vars, len);
  do_warmup(exec.cuda, alloc.managed, tm, num_vars, len);

  do_warmup(exec.seq,        alloc.managed_host_preferred, tm, num_vars, len);
  do_warmup(exec.cuda_batch, alloc.managed_host_preferred, tm, num_vars, len);

  do_warmup(exec.seq,             alloc.managed_host_preferred_device_accessed, tm, num_vars, len);
  do_warmup(exec.cuda_persistent, alloc.managed_host_preferred_device_accessed, tm, num_vars, len);

  {
    SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

    do_warmup(exec.seq,        alloc.managed_device_preferred, tm, num_vars, len);
    do_warmup(exec.cuda_batch, alloc.managed_device_preferred, tm, num_vars, len);

    do_warmup(exec.seq,             alloc.managed_device_preferred_host_accessed, tm, num_vars, len);
    do_warmup(exec.cuda_persistent, alloc.managed_device_preferred_host_accessed, tm, num_vars, len);
  }
#endif

#ifdef COMB_ENABLE_CUDA_GRAPH
  do_warmup(exec.cuda_graph, alloc.device, tm, num_vars, len);
#endif
}

} // namespace COMB
