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
    int numNeighbors;
    DataNode** neighbors;
    bool ghost;
    T data;
    int x,y,z; //spatial coordinates are only relavent in some networks
};

template <typename T> class ProcessNode {
    public:
    void bcastGPTRs();
    void gatherGhosts();

    //TODO: Come back and figure out which fields are safe to make private
    public:
    //data fields are public because the expectation
    //is that the data will interact with various kernels,
    //so it is exposed for easy access
    upcxx::global_ptr<DataNode<T>> m_data;

    //Communication example from process a:
    //story my data for neighbor i into m_sendData[i];
    //from neighbor i, a call is initiated where j is the idx process i according to process j's list of neighbors:
    //copy( a's m_sendData which is stored in process i in m_neighborSendData, m_neighborRecvData[j] )
    //unpack m_neighborRecvData[j] -> m_data using bufferMaps
    std::vector<upcxx::global_ptr<DataNode<T>>> m_neighborSendData; //global_ptrs to neighbor data
    std::vector<DataNode<T>*> m_neighborRecvData; //copy to here and unpack
    std::vector<upcxx::global_ptr<DataNode<T>>> m_sendData; //pack here (neighborSendData according to other processes)

    //we are only calling copy from receiving processes, so only need these buffer sizes tracked
    std::vector<size_t> m_neighborRecvSizes;

    //maps used for unpacking received data
    std::vector<std::vector<int>> m_bufferMaps;

    //tracking neighborhood in "process space"
    unsigned int m_numNeighbors;
    std::vector<int> m_neighborIDs;

    //we assume the following are initialized prior to a call to bcastGPTRs():
    //m_data, m_neighborSendData (to null pointers),
    //m_neighborRecvData (to empty arrays of correct size),
    //m_sendData (to empty arrays of correct size)
    
    //and the following initialized with accurate topology data
    //m_neighborRecvSizes, bufferMaps, numNeighbors, neighborIDs

};

template <typename T> void ProcessNode<T>::bcastGPTRs(){
    //send sendData global_pointer to neighbor processes' neighborSendData fields
    for(unsigned int i = 0; i < m_numNeighbors; i++){
        upcxx::rpc(m_neighborIDs[i],
                    [&](upcxx::global_ptr<DataNode<T>> gptr, int sourceRank){
                auto it = std::find(this->m_neighborIDs.begin(),
                                    this->m_neighborIDs.end(), sourceRank);
                int j = it - this->m_neighborIDs.begin();   
                this->m_neighborSendData[j] = gptr;
                std::cout << "rank " << upcxx::rank_me() << " knows that neighbor " << j << " is rank " << sourceRank << "\n";
            }, m_sendData[i], upcxx::rank_me()).wait();
    }
    upcxx::barrier();
}

template <typename T> void ProcessNode<T>::gatherGhosts(){
    upcxx::barrier(); //ensures all processes have packed data into sendData

    //receive data from neighbors
    upcxx::future<> futureAll = upcxx::make_future();
    for(unsigned int i = 0; i < m_numNeighbors; i++){
        upcxx::future<> f = upcxx::copy(m_neighborSendData[i], m_neighborRecvData[i], m_neighborRecvSizes[i]);
        futureAll = upcxx::when_all(futureAll, f);
    }
    futureAll.wait();

    //distribute data to correct DataNodes
    auto localData = m_data.local();
    for(unsigned int i = 0; i < m_numNeighbors; i++){
        for(unsigned int j = 0; j < m_bufferMaps[i].size(); j++){
            int localIDX = m_bufferMaps[i][j];
            localData[localIDX] = m_neighborRecvData[i][j];
            localData[localIDX].ghost = true;
        }
    }
    upcxx::barrier();
}
