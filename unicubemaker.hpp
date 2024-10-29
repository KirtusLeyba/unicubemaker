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

//TODO: 10/04/24, for some setups it is helpful to do communication in waves, and so there needs to be some way
//to do copies in a specific order from the application side

#pragma once
#include <vector>
#include <unordered_map>
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
    void recvAndUnpack();
    void packData();

    public:
    DataNode<T>* m_data_nodes;

    std::unordered_map<int, upcxx::global_ptr<DataNode<T>>> m_packed_data;
    std::unordered_map<int, size_t> m_packed_data_sizes;
    std::unordered_map<int, upcxx::global_ptr<DataNode<T>>> m_neighbor_data;
    std::unordered_map<int, size_t> m_neighbor_data_sizes;
    std::unordered_map<int, std::vector<int>> m_unpack_map;
    std::unordered_map<int, std::vector<int>> m_pack_map;
};

template <typename T> void ProcessNode<T>::bcastGPTRs(){
    for(auto pair : m_packed_data){
        int process_id = pair.first;
        upcxx::rpc(process_id,
                    [&](upcxx::global_ptr<DataNode<T>> gptr, int source_rank, size_t data_size){
                this->m_neighbor_data[source_rank] = gptr;
                this->m_neighbor_data_sizes[source_rank] = data_size;
            }, m_packed_data.at(process_id), upcxx::rank_me(), m_packed_data_sizes.at(process_id)).wait();
    }
}

template <typename T> void ProcessNode<T>::packData(){
    for(auto pair : m_pack_map){
        int process_id = pair.first;
        DataNode<T>* local_packed_data = m_packed_data.at(process_id).local();
        for(unsigned int i = 0; i < pair.second.size(); i++){
            local_packed_data[i].m_data = m_data_nodes[pair.second[i]].m_data;
        }
    }
}

template <typename T> void ProcessNode<T>::recvAndUnpack(){
    std::unordered_map<int, upcxx::global_ptr<DataNode<T>>> recv_data;
    
    //receive data from neighbors
    upcxx::future<> future_all = upcxx::make_future();
    for(auto pair : m_neighbor_data){
        int process_id = pair.first;
        upcxx::global_ptr<DataNode<T>> temp_recv = upcxx::new_array<DataNode<T>>(m_neighbor_data_sizes.at(process_id));
        upcxx::future<> f = upcxx::copy(m_neighbor_data.at(process_id), temp_recv, m_neighbor_data_sizes.at(process_id));
        future_all = upcxx::when_all(future_all, f);
        recv_data[process_id] = temp_recv;
    }
    future_all.wait();

    //unpack recv_data
    for(auto pair : m_unpack_map){
        int process_id = pair.first;
        for(unsigned int i = 0; i < m_neighbor_data_sizes.at(process_id); i++){
            m_data_nodes[pair.second[i]].m_data = recv_data.at(process_id).local()[i].m_data;
        }
    }

    //clear temp memory
    for(auto pair : recv_data){
        upcxx::delete_array(pair.second);
    }
}
