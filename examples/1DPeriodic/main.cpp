
#include "unicubemaker.hpp"

#include <vector>
#include <upcxx/upcxx.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

int main(int argc, char** argv){
    upcxx::init();

    //We are using a 1D simulation space with periodic boundaries.
    int num_cells = 100;
    int num_local_cells = num_cells / upcxx::rank_n();
    int num_cells_per_rank = num_cells / upcxx::rank_n();
    if(upcxx::rank_me() == upcxx::rank_n() - 1){
        num_local_cells += num_cells % upcxx::rank_n();
    }
    //account for ghost cells
    num_local_cells += 2;
    num_cells_per_rank += 2;

    //initialize our local process node
    ProcessNode<float> proc_node;
    proc_node.m_data = new DataNode<float>[num_local_cells];

    //set up the local data node network
    for(int i = 0; i < num_local_cells; i++){
        if(i == 0 || i == num_local_cells - 1){
            proc_node.m_data[i].m_num_neighbors = 1;
            proc_node.m_data[i].m_neighbors = new DataNode<float>*[1];
            if(i == 0) proc_node.m_data[i].m_neighbors[0] = &proc_node.m_data[1];
            else proc_node.m_data[i].m_neighbors[0] = &proc_node.m_data[num_local_cells - 2];
            proc_node.m_data[i].m_ghost = true;
            proc_node.m_data[i].m_data = 0.0f;
        } else {
            proc_node.m_data[i].m_num_neighbors = 2;
            proc_node.m_data[i].m_neighbors = new DataNode<float>*[2];
            int leftIDX = i - 1;
            int rightIDX = i + 1;
            proc_node.m_data[i].m_neighbors[0] = &proc_node.m_data[leftIDX];
            proc_node.m_data[i].m_neighbors[1] = &proc_node.m_data[rightIDX];
            proc_node.m_data[i].m_data = 0.0f;
            proc_node.m_data[i].m_ghost = false;
        }
        int worldX = i + upcxx::rank_me()*(num_cells_per_rank) - 1 - 2*(upcxx::rank_me());
        if(worldX == 5 && !proc_node.m_data[i].m_ghost) proc_node.m_data[i].m_data = 10000.0f;
    }

    //set up this process node with a left and right neighbor
    proc_node.m_neighbor_gptrs.push_back(NULL);
    proc_node.m_neighbor_gptrs.push_back(NULL);
    
    proc_node.m_packed_gptrs.push_back(upcxx::new_array<DataNode<float>>(1));
    proc_node.m_packed_gptrs.push_back(upcxx::new_array<DataNode<float>>(1));
    
    proc_node.m_neighbor_gptr_sizes.push_back(1);
    proc_node.m_neighbor_gptr_sizes.push_back(1);

    proc_node.m_gptr_to_data.push_back(std::vector<int>());
    proc_node.m_gptr_to_data.push_back(std::vector<int>());
    proc_node.m_gptr_to_data[0].push_back(0);
    proc_node.m_gptr_to_data[1].push_back(num_local_cells - 1);
    
    proc_node.m_num_neighbors = 2;
    
    proc_node.m_neighbor_ranks.push_back( upcxx::rank_me() - 1 );
    if(proc_node.m_neighbor_ranks[0] == -1) proc_node.m_neighbor_ranks[0] = upcxx::rank_n() - 1;
    proc_node.m_neighbor_ranks.push_back( upcxx::rank_me() + 1 );
    if(proc_node.m_neighbor_ranks[1] == upcxx::rank_n()) proc_node.m_neighbor_ranks[1] = 0;

    upcxx::barrier(); //setup finished

    //broadcast global pointers to establish the comms network
    proc_node.bcastGPTRs();

    //write output to file
    std::stringstream outputFileStream;
    outputFileStream << "./output/" << upcxx::rank_me() << "_log.csv";
    std::ofstream outputFile;
    outputFile.open(outputFileStream.str());
    
    outputFile << "it,x,amount,ghost" << std::endl;
    //run simulation
    int num_steps = 10000;
    float D = 0.1;
    for(int it = 0; it < num_steps; it++){
        DataNode<float>* new_data = new DataNode<float>[num_local_cells];
        //diffusion
        for(int i = 0; i < num_local_cells; i++){
            DataNode<float>* dn = &(proc_node.m_data[i]);
            if (dn->m_ghost) continue;
            new_data[i].m_data = dn->m_data;
            for(int j = 0; j < dn->m_num_neighbors; j++){
                new_data[i].m_data += D*dn->m_neighbors[j]->m_data;
            }
            new_data[i].m_data -= D*(float)dn->m_num_neighbors*dn->m_data;
        }
        //swap vectors
        for(int i = 0; i < num_local_cells; i++){
            proc_node.m_data[i].m_data = new_data[i].m_data;
            int worldX = i + upcxx::rank_me()*(num_cells_per_rank) - 1 - 2*(upcxx::rank_me());
            outputFile << it << "," << worldX << "," << proc_node.m_data[i].m_data << "," << proc_node.m_data[i].m_ghost << std::endl;
        }
        delete[] new_data;

        proc_node.m_packed_gptrs[0].local()[0] = proc_node.m_data[1];
        proc_node.m_packed_gptrs[1].local()[0] = proc_node.m_data[num_local_cells - 2];
        proc_node.gatherGhosts();
    }
    outputFile.close();

    //memory clean up
    delete[] proc_node.m_data;

    upcxx::finalize();
    return 0;
}
