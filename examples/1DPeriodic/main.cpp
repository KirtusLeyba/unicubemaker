
#include "unicubemaker.hpp"

#include <vector>
#include <upcxx/upcxx.hpp>

int main(int argc, char** argv){
    upcxx::init();

    //We are using a 1D simulation space with periodic boundaries.
    int numCells = 100;
    int numLocalCells = numCells / upcxx::rank_n();
    if(upcxx::rank_me() == upcxx::rank_n() - 1){
        numLocalCells += numCells % upcxx::rank_n();
    }
    //account for ghost cells
    numLocalCells += 2;

    //initialize our local process node
    ProcessNode<float> procNode;
    procNode.m_data = upcxx::new_array<DataNode<float>>(numLocalCells);
    //set up the local data node network
    DataNode<float>* localData = procNode.m_data.local();
    for(int i = 0; i < numLocalCells; i++){
        if(i == 0 || i == numLocalCells - 1){
            localData[i].numNeighbors = 1;
            localData[i].neighbors = new DataNode<float>*[1];
            if(i == 0) localData[i].neighbors[0] = &localData[1];
            else localData[i].neighbors[0] = &localData[numLocalCells - 2];
            localData[i].ghost = true;
            localData[i].data = 0.0f;
        } else {
            localData[i].numNeighbors = 2;
            localData[i].neighbors = new DataNode<float>*[2];
            int leftIDX = i - 1;
            int rightIDX = i + 1;
            localData[i].neighbors[0] = &localData[leftIDX];
            localData[i].neighbors[1] = &localData[rightIDX];
        }
        localData[i].x = i + upcxx::rank_me()*(numCells / upcxx::rank_n()) - 1;
        //setting initial concentration for diffusion
        if(localData[i].x == 49 && !localData[i].ghost) localData[i].data = 100.0f;
    }

    procNode.m_neighborSendData.push_back(NULL);
    procNode.m_neighborSendData.push_back(NULL);
    
    procNode.m_neighborRecvData.push_back(new DataNode<float>[1]);
    procNode.m_neighborRecvData.push_back(new DataNode<float>[1]);
    
    procNode.m_sendData.push_back(upcxx::new_array<DataNode<float>>(1));
    procNode.m_sendData.push_back(upcxx::new_array<DataNode<float>>(1));
    
    procNode.m_neighborRecvSizes.push_back(1);
    procNode.m_neighborRecvSizes.push_back(1);

    procNode.m_bufferMaps.push_back(std::vector<int>());
    procNode.m_bufferMaps.push_back(std::vector<int>());
    procNode.m_bufferMaps[0].push_back(0);
    procNode.m_bufferMaps[1].push_back(numLocalCells - 1);
    
    procNode.m_numNeighbors = 2;
    
    procNode.m_neighborIDs.push_back( upcxx::rank_me() - 1 );
    if(procNode.m_neighborIDs[0] == -1) procNode.m_neighborIDs[0] = upcxx::rank_n() - 1;
    procNode.m_neighborIDs.push_back( upcxx::rank_me() + 1 );
    if(procNode.m_neighborIDs[1] == upcxx::rank_n()) procNode.m_neighborIDs[1] = 0;

    //broadcast global pointers to establish the comms network
    procNode.bcastGPTRs();

    
    //run simulation
    int numSteps = 100;
    float D = 0.01;
    float* newData = new float[numLocalCells];
    for(int it = 0; it < numSteps; it++){


        //diffusion
        for(int i = 0; i < numLocalCells; i++){
            DataNode<float>* dn = &(procNode.m_data.local()[i]);
            if (dn->ghost) continue;
            newData[i] = dn->data;
            for(int j = 0; j < dn->numNeighbors; j++){
                newData[i] += D*dn->neighbors[j]->data;
            }
            newData[i] -= D*(float)dn->numNeighbors*dn->data;
        }
        for(int i = 0; i < numLocalCells; i++){
            DataNode<float>* dn = &(procNode.m_data.local()[i]);
            dn->data = newData[i];
            if(!dn->ghost){
                std::cout << it << "," << dn->x << "," << dn->data << "\n";
            }
        }

        //pack data
        //left neighbor
        procNode.m_sendData[0].local()[0] = procNode.m_data.local()[1];
        procNode.m_sendData[1].local()[1] = procNode.m_data.local()[numLocalCells - 2];

        procNode.gatherGhosts();
    }
    delete[] newData;


    //clean up memory
//    delete[] procNode.m_neighborRecvData[0];
//    delete[] procNode.m_neighborRecvData[1];

    for(int i = 0; i < numLocalCells; i++){
        if(procNode.m_data.local()[i].ghost) continue;
   //     delete[] procNode.m_data.local()[i].neighbors;
    }
    //upcxx::delete_array(procNode.m_data);
    //upcxx::delete_array(procNode.m_sendData[0]);
    //upcxx::delete_array(procNode.m_sendData[1]);

    upcxx::finalize();
    return 0;
}
