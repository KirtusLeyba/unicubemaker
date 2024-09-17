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

//** TODO: 05/06/2024 it looks like FCC will need its own special implementation because (with 1 cell thick boundary)
//**                  each process will have a different shape ghost buffer and whatnot

#pragma once
#include <vector>
#include <algorithm>
#include <upcxx/upcxx.hpp>

template <typename T> struct DataNode {
    int m_num_neighbors;
    DataNode** m_neighbors;
    bool m_ghost;
    T m_data;
};

template <typename T> class ProcessNode {
    public:
    void bcastGPTRs();
    void gatherGhosts();

    public:
    DataNode<T>* m_data;
    std::vector<upcxx::global_ptr<DataNode<T>>> m_packed_gptrs;
    std::vector<upcxx::global_ptr<DataNode<T>>> m_neighbor_gptrs;
    std::vector<size_t> m_neighbor_gptr_sizes;
    std::vector<std::vector<int>> m_gptr_to_data;

    unsigned int m_num_neighbors;
    std::vector<int> m_neighbor_ranks;

};

template <typename T> void ProcessNode<T>::bcastGPTRs(){
    upcxx::barrier();
    for(unsigned int i = 0; i < m_num_neighbors; i++){
        upcxx::rpc(m_neighbor_ranks[i],
                    [&](upcxx::global_ptr<DataNode<T>> gptr, int source_rank){
                auto it = std::find(this->m_neighbor_ranks.begin(),
                                    this->m_neighbor_ranks.end(), source_rank);
                int j = it - this->m_neighbor_ranks.begin();   
                this->m_neighbor_gptrs[j] = gptr;
            }, m_packed_gptrs[i], upcxx::rank_me()).wait();
    }
    upcxx::barrier();
}

template <typename T> void ProcessNode<T>::gatherGhosts(){
    upcxx::barrier(); //ensures all processes have packed data into sendData

    std::vector<DataNode<T>*> recv_data;

    //receive data from neighbors
    upcxx::future<> future_all = upcxx::make_future();
    for(unsigned int i = 0; i < m_num_neighbors; i++){

        recv_data.push_back(new DataNode<T>[m_neighbor_gptr_sizes[i]]);
        upcxx::future<> f = upcxx::copy(m_neighbor_gptrs[i], recv_data[i], m_neighbor_gptr_sizes[i]);
        future_all = upcxx::when_all(future_all, f);
    }
    future_all.wait();

    //distribute data to correct DataNodes
    for(unsigned int i = 0; i < m_num_neighbors; i++){
        for(unsigned int j = 0; j < m_gptr_to_data[i].size(); j++){
            m_data[m_gptr_to_data[i][j]].m_data = recv_data[i][j].m_data;
        }
    }

    //clear temp memory
    for(unsigned int i = 0; i < m_num_neighbors; i++){
        delete[] recv_data[i];
    }

    upcxx::barrier();
}
