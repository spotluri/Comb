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

#ifndef _UTILS_MP_HPP
#define _UTILS_MP_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_MP

#include <mp.h>

#include <cassert>
#include <cstdio>

#include "utils.hpp"
#include "utils_mpi.hpp"

namespace detail {

namespace mp {

MPI_Comm mp_comm;
int mp_device;

inline int init(MPI_Comm mpi_comm)
{
  cudaError_t err = cudaSuccess;
  
  mp_comm = mpi_comm;
  err = cudaGetDevice(&mp_device);
  assert(err == cudaSuccess);

  return 0;
}

inline int connect_ranks(std::set<int> ranks)
{
  int size = ranks.size; 
  int peers[size];

  for (int i=0; i<n; i++) { 
     peers[i] = ranks[i];
  }
  int res = mp_init(mp_comm, peers, size, MP_INIT_DEFAULT, mp_device);
  assert(res == 0); 
  return res;
}

inline void finalize()
{
  mp_finalize();
}

inline int register_region(void* ptr, size_t size, mp_reg_t *reg_t)
{
  int res = mp_register_region(ptr, size, reg_t);
  return res;
}

inline int deregister_region(mp_reg_t *reg)
{
  int res = mp_deregister_region(reg);
  return res;
}

inline void receive(void *buf, int size, int peer, mp_reg_t *mp_reg, mp_request_t *req)
{
  mpi_irecv(buf, size, peer, mp_reg, req);
}

inline int prepare_send(void *buf, int size, int peer, mp_ret_t *mp_reg, mp_request_t *req)
{
  mp_send_prepare(buf, size, peer, mp_reg, req);
}

inline int stream_post_isend(mp_request_t *req, cudaStream_t stream)
{
  mp_isend_post_on_stream(req, stream);
}

inline int stream_post_isend_all (uint32_t count, mp_request_t *req, cudaStream_t stream)
{
  mp_isend_post_all_on_stream(count, req, stream);
}

inline void stream_isend(void *buf, int size, int peer, mp_ret_t *mp_reg, mp_request_t *req, 
			cudaStream_t stream)
{
  mp_isend_on_stream(buf, size, peer, mp_reg, req, stream);
}

inline void isend(void *buf, int size, int peer, mp_ret_t *mp_reg, mp_request_t *req)
{
  mp_isend(buf, size, peer, mp_reg, req);
}

inline void stream_wait(mp_request_t *req, cudaStream_t stream)
{
  mp_wait_on_stream(req, stream);
}

inline int is_complete(mp_request_t *req)
{
  //mp does not have a test call
  return 0;
}

inline void wait(mp_request_t *req)
{
  mp_wait(req);
}

inline void wait_all(int count, mp_request_t *req)
{
  mp_wait_all(count, req);
}

} // namespace mp

} // namespace detail

#endif // COMB_ENABLE_MP

#endif // _UTILS_MP_HPP

