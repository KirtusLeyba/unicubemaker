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

#include <vector>
#include <algorithm>
#include <upcxx/upcxx.hpp>

template <typename T> struct DataNode {
    int numNeighbors;
    DataNode* neighbors;
    bool ghost;
    T data;
    int x,y,z; //spatial coordinates are only relavent in some networks
};

template <typename T> class ProcessNode {
    public:
    void initComms();
    void gatherGhosts();

    public:
    //data fields are public because the expectation
    //is that the data will interact with various kernels,
    //so it is exposed for easy access
    upcxx::global_ptr<DataNode<T>> data;
    std::vector<upcxx::global_ptr<DataNode<T>>> neighborSendData; //global_ptrs to neighbor data
    std::vector<upcxx::global_ptr<DataNode<T>>> neighborRecvData; //copy to here and unpack
    std::vector<upcxx::global_ptr<DataNode<T>>> sendData; //pack here (neighborSendData according to other processes)
    std::vector<size_t> neighborBufferSizes;

    //maps used for unpacking received data
    std::vector<std::vector<int>> bufferMaps;
    
    private:
    unsigned int numNeighbors;
    std::vector<int> neighborIDs;

};

void ProcessNode::initComms(){
    //send sendData global_pointer to neighbor processes' neighborSendData fields
    for(unsigned int i = 0; i < numNeighbors; i++){
        upcxx::rpc(neighborIDs[i],
                    [&](upcxx::global_ptr<DataNode<T>> gptr, int sourceRank){
                auto it = std::find(this->neighborIDs.begin(),
                                    this->neighborIDs.end(), sourceRank);
                int j = *it;   
                this->neighborSendData[j] = gptr; //NOTE: no push_back, so must be initialized at this point
            }, sendData[i], upcxx::rank_me()).wait();
    }
    upcxx::barrier();
}

void ProcessNode::gatherGhosts(){
    upcxx::barrier(); //ensures all processes have packed data into sendData

    //receive data from neighbors
    upcxx::future<> futureAll = upcxx::make_future();
    for(unsigned int i = 0; i < numNeighbors; i++){
        upcxx::future<> f = upcxx::copy(neighborSendData[i], neighborRecvData[i], neighborBufferSizes[i]);
        futureAll = upcxx::when_all(futureAll, f);
    }
    futureAll.wait();

    //distribute data to correct DataNodes
    auto localData = data.local();
    for(unsigned int i = 0; i < numNeighbors; i++){
        for(unsigned int j = 0; j < bufferMaps[i].size(); j++){
            int localIDX = bufferMaps[i][j];
            localData[localIDX] = neighborRecvData[i][j];
        }
    }

    upcxx::barrier();
}
