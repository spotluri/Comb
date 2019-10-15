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

#ifndef _COMM_POL_MP_HPP
#define _COMM_POL_MP_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_MP

#include <exception>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <map>

#include "for_all.hpp"
#include "utils.hpp"
#include "utils_cuda.hpp"
#include "utils_mp.hpp"
#include "MessageBase.hpp"
#include "ExecContext.hpp"

struct mp_pol {
  // static const bool async = false;
  static const bool mock = false;
  // compile mpi_type packing/unpacking tests for this comm policy
  static const bool use_mpi_type = false;
  static const char* get_name() { return "mp"; }
  using send_request_type = mp_request_t;
  using recv_request_type = mp_request_t;
  using send_status_type = int;
  using recv_status_type = int;
};

template < >
struct CommContext<mp_pol> : CudaContext
{
  using base = CudaContext;

  using pol = mp_pol;

  using send_request_type = typename pol::send_request_type;
  using recv_request_type = typename pol::recv_request_type;
  using send_status_type = typename pol::send_status_type;
  using recv_status_type = typename pol::recv_status_type;

  MPI_Comm comm = MPI_COMM_NULL;

  CommContext()
    : base()
  { }

  CommContext(base const& b)
    : base(b)
  { 
  }

  CommContext(CommContext const& a_, MPI_Comm comm_)
    : base(a_)
    , comm(comm_)
  {
      detail::mp::finalize(comm_); 
  }

  ~CommContext()
  {
    if (comm != MPI_COMM_NULL) { 
      detail::mp::finalize(); 
      active = 0;
    }
  }

  void ensure_waitable()
  {

  }

  template < typename context >
  void waitOn(context& con)
  {
    con.ensure_waitable();
    base::waitOn(con);
  }

  send_request_type send_request_null() { return send_request_type{}; }
  recv_request_type recv_request_null() { return recv_request_type{}; }
  send_status_type send_status_null() { return 0; }
  recv_status_type recv_status_null() { return 0; }


  void connect_ranks(std::vector<int> const& send_ranks,
                     std::vector<int> const& recv_ranks,
		     MPI_Comm mpi_comm)
  {
    std::set<int> ranks;
    for (int rank : send_ranks) {
      if (ranks.find(rank) == ranks.end()) {
        ranks.insert(rank);
      }
    }
    for (int rank : recv_ranks) {
      if (ranks.find(rank) == ranks.end()) {
        ranks.insert(rank);
      }
    }

    detail::mp::connect_ranks(ranks);
  }

  void disconnect_ranks()
  {
    detail::mp::finalize();
  }
};

struct mp_mempool
{
  struct ibv_ptr
  {
    mp_reg_t mr;
    size_t offset = 0;
    void* ptr = nullptr;
  };

  ibv_ptr allocate(COMB::Allocator& aloc_in, size_t size)
  {
    ibv_ptr ptr{};
    int status = 0;
    if (size > 0) {
      size = std::max(size, sizeof(std::max_align_t));

      auto iter = m_allocators.find(&aloc_in);
      if (iter != m_allocators.end()) {
        COMB::Allocator& aloc = *iter->first;
        used_ptr_map&    used_ptrs = iter->second.used;
        unused_ptr_map&  unused_ptrs = iter->second.unused;

        ptr_info info{};

        auto unused_iter = unused_ptrs.find(size);
        if (unused_iter != unused_ptrs.end()) {
          // found an existing unused ptr
          info = unused_iter->second;
          unused_ptrs.erase(unused_iter);
          used_ptrs.emplace(info.ptr.ptr, info);
        } else {
          // allocate a new pointer for this size
          info.size = size;
          info.ptr.ptr = aloc.allocate(info.size);
          info.ptr.offset = 0;
          status = detail::mp::register_region(info.ptr.ptr, info.size, &info.ptr.mr);
	  assert(status == 0);
          used_ptrs.emplace(info.ptr.ptr, info);
        }

        ptr = info.ptr;
      } else {
        throw std::invalid_argument("unknown allocator passed to mp_mempool::allocate");
      }
    }
    return ptr;
  }

  void deallocate(COMB::Allocator& aloc_in, ibv_ptr ptr)
  {
    if (ptr.ptr != nullptr) {

      auto iter = m_allocators.find(&aloc_in);
      if (iter != m_allocators.end()) {
        COMB::Allocator& aloc = *iter->first;
        used_ptr_map&    used_ptrs = iter->second.used;
        unused_ptr_map&  unused_ptrs = iter->second.unused;

        auto used_iter = used_ptrs.find(ptr.ptr);
        if (used_iter != used_ptrs.end()) {
          // found an existing used ptr
          ptr_info info = used_iter->second;
          used_ptrs.erase(used_iter);
          unused_ptrs.emplace(info.size, info);
        } else {
          // unknown or unused pointer
          throw std::invalid_argument("unknown or unused pointer passed to mp_mempool::deallocate");
        }
      } else {
        throw std::invalid_argument("unknown allocator passed to mp_mempool::deallocate");
      }
    }
  }

  void add_allocator(COMB::Allocator& aloc)
  {
    if (m_allocators.find(&aloc) == m_allocators.end()) {
      // new allocator
      m_allocators.emplace(&aloc, ptr_map{});
    }
  }

  void remove_allocators()
  {
    bool error = false;
    auto iter = m_allocators.begin();
    while (iter != m_allocators.end()) {
      COMB::Allocator& aloc = *iter->first;
      used_ptr_map&    used_ptrs = iter->second.used;
      unused_ptr_map&  unused_ptrs = iter->second.unused;

      auto inner_iter = unused_ptrs.begin();
      while (inner_iter != unused_ptrs.end()) {
        ptr_info& info = inner_iter->second;

        detail::mp::deregister_region(&info.ptr.mr);
        aloc.deallocate(info.ptr.ptr);
        inner_iter = unused_ptrs.erase(inner_iter);
      }

      if (used_ptrs.empty()) {
        iter = m_allocators.erase(iter);
      } else {
        ++iter;
        error = true;
      }
    }

    if (error) throw std::logic_error("can not remove Allocator with used ptr");
  }

private:
  struct ptr_info
  {
    ibv_ptr ptr{};
    size_t size = 0;
  };
  using used_ptr_map = std::unordered_map<void*, ptr_info>;
  using unused_ptr_map = std::multimap<size_t, ptr_info>;
  struct ptr_map
  {
    used_ptr_map used{};
    unused_ptr_map unused{};
  };

  std::unordered_map<COMB::Allocator*, ptr_map> m_allocators;
};


template < >
struct Message<mp_pol> : detail::MessageBase
{
  using base = detail::MessageBase;

  using policy_comm = mp_pol;
  using communicator_type = CommContext<policy_comm>;
  using send_request_type = typename policy_comm::send_request_type;
  using recv_request_type = typename policy_comm::recv_request_type;
  using send_status_type  = typename policy_comm::send_status_type;
  using recv_status_type  = typename policy_comm::recv_status_type;

  using region_type = mp_mempool::ibv_ptr;
  static inline mp_mempool& get_mempool()
  {
    static mp_mempool mempool;
    return mempool;
  }

  static void setup_mempool(
                            COMB::Allocator& many_aloc,
                            COMB::Allocator& few_aloc)
  {
    get_mempool().add_allocator(many_aloc);
    get_mempool().add_allocator(few_aloc);
  }

  static void teardown_mempool()
  {
    get_mempool().remove_allocators();
  }


  Message(Kind _kind, int partner_rank, int tag, bool have_many)
    : base(_kind, partner_rank, tag, have_many)
    , m_region()
  { }

  ~Message()
  { }

  template < typename context >
  void pack(context& con, communicator_type& con_comm)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "mp_pol does not support mpi_type_pol");
    DataT* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT const* src = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      // FGPRINTF(FileGroup::proc, "%p pack %p = %p[%p] len %d\n", this, buf, src, indices, len);
      con.for_all(0, len, make_copy_idxr_idxr(src, detail::indexer_list_idx{indices}, buf, detail::indexer_idx{}));
      buf += len;
    }
  }

  template < typename context >
  void unpack(context& con, communicator_type& con_comm)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "mp_pol does not support mpi_type_pol");
    DataT const* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT* dst = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      // FGPRINTF(FileGroup::proc, "%p unpack %p[%p] = %p len %d\n", this, dst, indices, buf, len);
      con.for_all(0, len, make_copy_idxr_idxr(buf, detail::indexer_idx{}, dst, detail::indexer_list_idx{indices}));
      buf += len;
    }
  }

  template < typename context >
  void allocate(context&, communicator_type& con_comm, COMB::Allocator& buf_aloc)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "mp_pol does not support mpi_type_pol");
    if (m_buf == nullptr) {
      m_region = get_mempool().allocate(con_comm.g, buf_aloc, nbytes());
      m_buf = (DataT*)m_region.ptr;
    }
  }

  template < typename context >
  void deallocate(context& con, communicator_type& con_comm, COMB::Allocator& buf_aloc)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "mp_pol does not support mpi_type_pol");
    if (m_buf != nullptr) {

      if (m_send_request) {
        finish_send(con_comm, *m_send_request);
        m_send_request = nullptr;
      }
      if (m_recv_request) {
        finish_recv(con_comm, *m_recv_request);
        m_recv_request = nullptr;
      }
      get_mempool().deallocate(con_comm.g, buf_aloc, m_region);
      m_region = region_type{};
      m_buf = nullptr;
    }
  }

private:
  void start_Isend(CPUContext const&,  communicator_type& con_comm, send_request_type* request)
  {
    COMB::ignore_unused(con_comm);
  }

  void start_Isend(CudaContext const& con,  communicator_type& con_comm, send_request_type* request)
  {
    detail::mp::stream_isend(m_region.ptr, nbytes(), partner_rank(), m_region.mr, &request->req, con.stream);
  }

public:
  template < typename context >
  void Isend(context& con, communicator_type& con_comm, send_request_type* request)
  {
    start_Isend(con, con_comm, request);
    request->status = 1;
    request->setContext(con);
    request->completed = false;
    m_send_request = request;
  }

private:

  void prepare_send(CPUContext const&,  communicator_type& con_comm, send_request_type* request)
  {
    COMB::ignore_unused(con_comm);
  }

  void prepare_send(CudaContext const& con,  communicator_type& con_comm, send_request_type* request)
  {
    detail::mp::prepare_send(m_region.ptr, nbytes(), partner_rank(), m_region.mr, request);
  }

public:
  template < typename context >
  void prepare_Isend(context& con, communicator_type& con_comm, send_request_type* request)
  {
    prepare_send(con, con_comm, request);
    m_send_request = request;
  }

  void post_Isend(CPUContext const&,  communicator_type& con_comm, send_request_type* request)
  {
    COMB::ignore_unused(con_comm);
  }

  void post_Isend(CudaContext const& con,  communicator_type& con_comm, send_request_type* request)
  {
    detail::mp::stream_post_isend(request, con.stream);
  }

  void post_Isend_all(CPUContext const&,  communicator_type& con_comm, int count, send_request_type* request)
  {
    COMB::ignore_unused(con_comm);
  }

  void post_Isend_all(CudaContext const& con, communicator_type& con_comm, int count, send_request_type* request)
  {
    detail::mp::stream_post_isend_all(count, request, con.stream);
  }

  template < typename context >
  void Irecv(context& con, communicator_type& con_comm, recv_request_type* request)
  {
    detail::mp::receive(m_region.ptr, nbytes(), partner_rank(), m_region.mr, request);
    m_recv_request = request;
  }

private:
  static bool start_wait_send(context& con, communicator_type& con_comm,
                              send_request_type& request)
  {
    if (request.context_type == ContextEnum::cuda) {
      detail::mp::stream_wait(request, request.context.cuda.stream());
    } else { 
      COMB::ignore_unused(con_comm);
    }
    return done;
  }

public:
  static void wait_send_all(communicator_type& con_comm,
                            int count, send_request_type* requests,
                            send_status_type* statuses)
  {
    detail::mp::mp_wait_all(count, requests);
  }

private:
  static bool start_wait_recv(communicator_type&,
                              recv_request_type& request)
  {
    if (request.context_type == ContextEnum::cuda) {
      detail::mp::stream_wait(request, request.context.cuda.stream());
    } else { 
      COMB::ignore_unused(con_comm);
    } else {
      assert(0 && (request.context_type == ContextEnum::cuda || request.context_type == ContextEnum::cpu));
    }
    return done;
  }

public:
  static void wait_recv_all(communicator_type& con_comm,
                            int count, recv_request_type* requests,
                            recv_status_type* statuses)
  {
    detail::mp::mp_wait_all(count, requests);
  }

private:
  region_type m_region;
  send_request_type* m_send_request = nullptr;
  recv_request_type* m_recv_request = nullptr;
};

#endif // COMB_ENABLE_MP

#endif // _COMM_POL_MP_HPP
