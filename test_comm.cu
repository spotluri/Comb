#include <cstdio>
#include <cstdlib>

#include <mpi.h>

#include "memory.cuh"
#include "for_all.cuh"
#include "profiling.cuh"
#include "mesh.cuh"
#include "comm.cuh"

namespace detail {

  struct set_n1 {
     DataT* data;
     set_n1(DataT* data_) : data(data_) {}
     HOST DEVICE
     void operator()(IdxT i, IdxT) const {
       IdxT zone = i;
       //printf("%p[%i] = %f\n", data, zone, 1.0); fflush(stdout);
       data[zone] = -1.0;
     }
  };

  struct set_1 {
     IdxT ilen, ijlen;
     DataT* data;
     set_1(IdxT ilen_, IdxT ijlen_, DataT* data_) : ilen(ilen_), ijlen(ijlen_), data(data_) {}
     HOST DEVICE
     void operator()(IdxT k, IdxT j, IdxT i, IdxT idx) const {
       IdxT zone = i + j * ilen + k * ijlen;
       //printf("%p[%i] = %f\n", data, zone, 1.0); fflush(stdout);
       data[zone] = 1.0;
     }
  };

  struct reset_1 {
     IdxT ilen, ijlen;
     DataT* data;
     IdxT imin, jmin, kmin;
     IdxT imax, jmax, kmax;
     reset_1(IdxT ilen_, IdxT ijlen_, DataT* data_, IdxT imin_, IdxT jmin_, IdxT kmin_, IdxT imax_, IdxT jmax_, IdxT kmax_)
       : ilen(ilen_), ijlen(ijlen_), data(data_)
       , imin(imin_), jmin(jmin_), kmin(kmin_)
       , imax(imax_), jmax(jmax_), kmax(kmax_)
     {}
     HOST DEVICE
     void operator()(IdxT k, IdxT j, IdxT i, IdxT idx) const {
       IdxT zone = i + j * ilen + k * ijlen;
       DataT expected, found, next;
       if (k >= kmin && k < kmax &&
           j >= jmin && j < jmax &&
           i >= imin && i < imax) {
         expected = 1.0; found = data[zone]; next = 1.0;
       } else {
         expected = 0.0; found = data[zone]; next = -1.0;
       }
       //if (found != expected) printf("zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
       //printf("%p[%i] = %f\n", data, zone, 1.0); fflush(stdout);
       data[zone] = next;
     }
  };

} // namespace detail

template < typename pol_loop, typename pol_face, typename pol_edge, typename pol_corner >
void do_cycles(MeshInfo& mesh, IdxT ncycles, Allocator& aloc_mesh, Allocator& aloc_face, Allocator& aloc_edge, Allocator& aloc_corner, Timer& tm)
{
    tm.clear();

    char rname[1024] = ""; snprintf(rname, 1024, "Buffers %s %s %s %s %s %s", pol_face::name, aloc_face.name(), pol_edge::name, aloc_edge.name(), pol_corner::name, aloc_corner.name());
    char test_name[1024] = ""; snprintf(test_name, 1024, "Mesh %s %s %s", pol_loop::name, aloc_mesh.name(), rname);
    printf("Starting test %s\n", test_name); fflush(stdout);

    Range r0(rname, Range::orange);

    tm.start("start-up");

    MeshData var(mesh, aloc_mesh);
    Comm<pol_face, pol_edge, pol_corner> comm(aloc_face, aloc_edge, aloc_corner);

    {
      var.allocate();
      
      DataT* data = var.data();
      IdxT ijklen = mesh.ijklen;

      for_all(pol_loop{}, 0, ijklen,
                          detail::set_n1(data));

      if (pol_loop::async) {cudaCheck(cudaDeviceSynchronize());}

    }
    
    {
      comm.add_var(var);
    }

    tm.stop();

    for(IdxT cycle = 0; cycle < ncycles; cycle++) {

      Range r1("cycle", Range::yellow);

      IdxT imin = mesh.imin;
      IdxT jmin = mesh.jmin;
      IdxT kmin = mesh.kmin;
      IdxT imax = mesh.imax;
      IdxT jmax = mesh.jmax;
      IdxT kmax = mesh.kmax;
      IdxT ilen = mesh.ilen;
      IdxT jlen = mesh.jlen;
      IdxT klen = mesh.klen;
      IdxT ijlen = mesh.ijlen;
      
      DataT* data = var.data();
      
      Range r2("pre-comm", Range::red);
      tm.start("pre-comm");

      for_all_3d(pol_loop{}, kmin, kmax,
                             jmin, jmax,
                             imin, imax,
                             detail::set_1(ilen, ijlen, data));

      if (pol_loop::async) {cudaCheck(cudaDeviceSynchronize());}

      tm.stop();
      r2.restart("post-recv", Range::pink);
      tm.start("post-recv");
      
      comm.postRecv();

      tm.stop();
      r2.restart("post-send", Range::pink);
      tm.start("post-send");

      comm.postSend();

      if (pol_corner::async || pol_edge::async || pol_face::async) {cudaCheck(cudaDeviceSynchronize());}
      
      tm.stop();
      r2.stop();
      
      /*
      for_all_3d(pol_loop{}, 0, klen,
                             0, jlen,
                             0, ilen,
                             [=] HOST DEVICE (IdxT k, IdxT j, IdxT i, IdxT idx) {
        IdxT zone = i + j * ilen + k * ijlen;
        DataT expected, found, next;
        if (k >= kmin && k < kmax &&
            j >= jmin && j < jmax &&
            i >= imin && i < imax) {
          expected = 1.0; found = data[zone]; next = 1.0;
        } else {
          expected = -1.0; found = data[zone]; next = -1.0;
        }
        if (found != expected) printf("zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
        //printf("%p[%i] = %f\n", data, zone, 1.0);
        data[zone] = next;
      });
      */

      r2.start("wait-recv", Range::pink);
      tm.start("wait-recv");

      comm.waitRecv();

      if (pol_corner::async || pol_edge::async || pol_face::async) {cudaCheck(cudaDeviceSynchronize());}

      tm.stop();
      r2.restart("wait-send", Range::pink);
      tm.start("wait-send");

      comm.waitSend();

      tm.stop();
      r2.restart("post-comm", Range::red);
      tm.start("post-comm");

      for_all_3d(pol_loop{}, 0, klen,
                             0, jlen,
                             0, ilen,
                             detail::reset_1(ilen, ijlen, data, imin, jmin, kmin, imax, jmax, kmax));

      if (pol_loop::async) {cudaCheck(cudaDeviceSynchronize());}

      tm.stop();
      r2.stop();

    }

    tm.print();
    tm.clear();
}
 

int main(int argc, char** argv)
{
  int required = MPI_THREAD_SINGLE;
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, required, &provided);

  MPI_Comm mpi_comm = MPI_COMM_WORLD;

  if (required != provided) {
    fprintf(stderr, "Didn't receive MPI thread support required %i provided %i.\n", required, provided); fflush(stderr);
    MPI_Abort(mpi_comm, 1);
  }

  int comm_rank = -1;
  MPI_Comm_rank(mpi_comm, &comm_rank);
  int comm_size = 0;
  MPI_Comm_size(mpi_comm, &comm_size);

  if (comm_rank == 0) {
    printf("Started rank %i of %i\n", comm_rank, comm_size); fflush(stdout);
  }

  cudaCheck(cudaDeviceSynchronize());  

  IdxT sizes[] = {0, 0, 0};
  IdxT ghost_width = 1;
  
  IdxT i = 1;
  IdxT s = 0;
  for(; i < argc; ++i) {
    if (argv[i][0] == '-') {
      // option
      if (strcmp(&argv[i][1], "ghost") == 0) {
        if (++i < argc) {
          ghost_width = static_cast<IdxT>(atoll(argv[i]));
        } else {
          fprintf(stderr, "No argument to option, ignoring %s.\n", argv[i-1]); fflush(stderr);
        }
      } else {
        fprintf(stderr, "Unknown option, ignoring %s.\n", argv[i]); fflush(stderr);
      }
    } else if ( s < 3 ) {
      // assume got integer
      sizes[s++] = static_cast<IdxT>(atoll(argv[i]));
    } else {
      fprintf(stderr, "Too many arguments, ignoring %s.\n", argv[i]); fflush(stderr);
    }
  }
  
  if (s == 1) {
    sizes[1] = sizes[0];
    sizes[2] = sizes[0];
  }
  
  if (ghost_width <= 0) {
    if (comm_rank == 0) {
      fprintf(stderr, "Invalid ghost argument.\n"); fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  } else if (sizes[0] < ghost_width || sizes[1] < ghost_width || sizes[2] < ghost_width) {
    if (comm_rank == 0) {
      fprintf(stderr, "Invalid size arguments.\n"); fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MeshInfo mesh(sizes[0], sizes[1], sizes[2], ghost_width);
    
  if (comm_rank == 0) {
    printf("Mesh info\n");
    printf("%i %i %i\n", mesh.isize, mesh.jsize, mesh.ksize);
    printf("ij %i ik %i jk %i\n", mesh.ijsize, mesh.iksize, mesh.jksize);
    printf("ijk %i\n", mesh.ijksize);
    printf("ghost_width %i\n", mesh.ghost_width);
    printf("i %8i %8i %8i %8i\n", 0, mesh.imin, mesh.imax, mesh.ilen);
    printf("j %8i %8i %8i %8i\n", 0, mesh.jmin, mesh.jmax, mesh.jlen);
    printf("k %8i %8i %8i %8i\n", 0, mesh.kmin, mesh.kmax, mesh.klen);
    printf("ij %i ik %i jk %i\n", mesh.ijlen, mesh.iklen, mesh.jklen);
    printf("ijk %i\n", mesh.ijklen);
    fflush(stdout);
  }
  
  HostAllocator host_alloc;
  HostPinnedAllocator hostpinned_alloc;
  DeviceAllocator device_alloc;
  ManagedAllocator managed_alloc;
  ManagedHostPreferredAllocator managed_host_preferred_alloc;
  ManagedHostPreferredDeviceAccessedAllocator managed_host_preferred_device_accessed_alloc;
  ManagedDevicePreferredAllocator managed_device_preferred_alloc;
  ManagedDevicePreferredHostAccessedAllocator managed_device_preferred_host_accessed_alloc;
  
  Timer tm(1024);

  // warm-up memory pools
  {
    printf("Starting up memory pools\n"); fflush(stdout);

    Range r("Memmory pool init", Range::green);

    void* var0;
    void* var1;
 
    tm.start(host_alloc.name());

    var0 = host_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = host_alloc.allocate(mesh.ijksize*sizeof(DataT));
    host_alloc.deallocate(var0);
    host_alloc.deallocate(var1);

    tm.restart(hostpinned_alloc.name());

    var0 = hostpinned_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = hostpinned_alloc.allocate(mesh.ijksize*sizeof(DataT));
    hostpinned_alloc.deallocate(var0);
    hostpinned_alloc.deallocate(var1);

    tm.restart(device_alloc.name());

    var0 = device_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = device_alloc.allocate(mesh.ijksize*sizeof(DataT));
    device_alloc.deallocate(var0);
    device_alloc.deallocate(var1);

    tm.restart(managed_alloc.name());

    var0 = managed_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = managed_alloc.allocate(mesh.ijksize*sizeof(DataT));
    managed_alloc.deallocate(var0);
    managed_alloc.deallocate(var1);

    tm.restart(managed_host_preferred_alloc.name());

    var0 = managed_host_preferred_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = managed_host_preferred_alloc.allocate(mesh.ijksize*sizeof(DataT));
    managed_host_preferred_alloc.deallocate(var0);
    managed_host_preferred_alloc.deallocate(var1);

    tm.restart(managed_host_preferred_device_accessed_alloc.name());

    var0 = managed_host_preferred_device_accessed_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = managed_host_preferred_device_accessed_alloc.allocate(mesh.ijksize*sizeof(DataT));
    managed_host_preferred_device_accessed_alloc.deallocate(var0);
    managed_host_preferred_device_accessed_alloc.deallocate(var1);

    tm.restart(managed_device_preferred_alloc.name());

    var0 = managed_device_preferred_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = managed_device_preferred_alloc.allocate(mesh.ijksize*sizeof(DataT));
    managed_device_preferred_alloc.deallocate(var0);
    managed_device_preferred_alloc.deallocate(var1);

    tm.restart(managed_device_preferred_host_accessed_alloc.name());

    var0 = managed_device_preferred_host_accessed_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = managed_device_preferred_host_accessed_alloc.allocate(mesh.ijksize*sizeof(DataT));
    managed_device_preferred_host_accessed_alloc.deallocate(var0);
    managed_device_preferred_host_accessed_alloc.deallocate(var1);

    tm.stop();

    tm.print();
    tm.clear();

  }

  IdxT ncycles = 5;

  // host allocated
  {
    Allocator& mesh_aloc = host_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);
  }

  // host pinned allocated
  {
    Allocator& mesh_aloc = hostpinned_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // device allocated
  {
    Allocator& mesh_aloc = device_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    // do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
 
    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(mesh, ncycles, mesh_aloc, device_alloc, device_alloc, device_alloc, tm);
  }

  // managed allocated
  {
    Allocator& mesh_aloc = managed_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, seq_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // managed host preferred allocated
  {
    Allocator& mesh_aloc = managed_host_preferred_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, seq_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // managed host preferred device accessed allocated
  {
    Allocator& mesh_aloc = managed_host_preferred_device_accessed_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, seq_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // managed device preferred allocated
  {
    Allocator& mesh_aloc = managed_device_preferred_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, seq_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // managed device preferred host accessed allocated
  {
    Allocator& mesh_aloc = managed_device_preferred_host_accessed_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol, seq_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, seq_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(mesh, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  MPI_Finalize();
  return 0;
}

