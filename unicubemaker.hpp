/**
* Copyright (C) Kirtus G. Leyba, 2024
* This software is freely available without warranty of any kind according
* to the MIT license found in the LICENSE file.
**/

/**
* UniCubeMaker is a header only library for subdividing simulation grids into smaller cubes.
* One of the common needs of discrete spatial simulations is distributing 3-dimensional
* grids of cells to multiple processes, and I have had to implement it often enough
* that I decided to make a portable library that can quickly be reused. For parallelism,
* UniCuberMaker uses upc++. See (https://bitbucket.org/berkeleylab/upcxx/wiki/Home)
*
* The point of this library is to be very straightforward to use and be easily extended or modified.
*
* The data structures use C++ templates which allows for easy reuse in different scenarios, but also
* restricts the linking of prebuilt static or dynamic libraries, so keep that in mind.
*
*/

#include <upcxx/upcxx.hpp>

struct Constraints {
    // the full dimensions of simulation space, IS NOT accounting for ghost voxels
    int fullX, fullY, fullZ, fullSize;
    // the local dimensions of simulation space, owned by this process, IS NOT accounting for ghost voxels
    int localX, localY, localZ, localSize;
    // the local dimensions of the simulation space, owned by this process, IS accounting for ghost voxels
    int memX, memY, memZ, memSize;
};

template <typename T> class MemoryHandler {
    public:
    MemoryHandler(Constraints cons);
    ~MemoryHandler();
    
    private:
    void allocate();
    void deallocate();

    public:
    //data fields are public because the expectation
    //is that the data will interact with various kernels,
    //so it is exposed for easy access
    upcxx::dist_object< upcxx::global_ptr<T> > distData;

    

    private:
        Constraints m_Cons;
};

template <typename T> MemoryHandler<T>::MemoryHandler(Constraints cons){
    m_Cons = cons;
}

template <typename T> MemoryHandler<T>::~MemoryHandler(){
    return; //TODO
}

template <typename T> void MemoryHandler<T>::allocate(){
    
}

template <typename T> void MemoryHandler<T>::allocate(){
    
}
