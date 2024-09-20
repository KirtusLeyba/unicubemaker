
#include "unicubemaker.hpp"

#include <vector>
#include <upcxx/upcxx.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <type_traits>

struct Point3D {
    int x;
    int y;
    int z;
};
Point3D to3D(int i, Point3D dims){
    int x = i / (dims.z*dims.y);
    int y = (i % (dims.z*dims.y)) / dims.z;
    int z  =(i % (dims.z*dims.y)) % dims.z;
    return {x, y, z};
}
int to1D(Point3D p, Point3D dims){
    return p.z + p.y*dims.z + p.x*(dims.y*dims.z);
}
bool inBounds(Point3D pos, Point3D dims){
    if(pos.x >= dims.x || pos.x < 0) return false;
    if(pos.y >= dims.y || pos.y < 0) return false;
    if(pos.z >= dims.z || pos.z < 0) return false;
    return true;
}

struct FCCDiffuser {
    int x;
    int y;
    int z;
    int i;
    float amount;
};

struct Rank {
    Point3D my_half_steps;
    Point3D coords_rank_space;
    Point3D coords_sim_space;
    bool prevX;
    bool nextX;
    bool prevY;
    bool nextY;
    bool prevZ;
    bool nextZ;
    int num_local_cells;
};

bool isPoint(bool evenX, bool evenY, bool evenZ){
    if(evenZ){
        if(evenY){
            if(evenX){
                return true;
            }
        } else {
            if(!evenX){
                return true;
            }
        }
    } else {
        if(evenY){
            if(!evenX){
                return true;
            }
        } else {
            if(evenX){
                return true;
            }
        }
    }
    return false;
}

int main(int argc, char** argv){
    upcxx::init();

    //Using FCC 3D with empty boundaries
    //global spatial information
    int half_steps_x = 7;
    int half_steps_y = 7;
    int half_steps_z = 7;
    int num_ranks = upcxx::rank_n();
    Point3D rank_dims = {2, 2, 2};

    Rank rank;
    rank.prevX = false;
    rank.nextX = false;
    rank.prevY = false;
    rank.nextY = false;
    rank.prevZ = false;
    rank.nextZ = false;
    Point3D half_steps_per_rank = { half_steps_x / rank_dims.x,
                                    half_steps_y / rank_dims.y,
                                    half_steps_z / rank_dims.z };
    rank.my_half_steps = {  half_steps_per_rank.x,
                            half_steps_per_rank.y,
                            half_steps_per_rank.z};
    rank.coords_rank_space = to3D(upcxx::rank_me(), rank_dims);
    rank.coords_sim_space = {rank.coords_rank_space.x*half_steps_per_rank.x,
                             rank.coords_rank_space.y*half_steps_per_rank.y,
                             rank.coords_rank_space.z*half_steps_per_rank.z};
    if(rank.coords_rank_space.x == rank_dims.x - 1) rank.my_half_steps.x += half_steps_x % rank_dims.x;
    if(rank.coords_rank_space.y == rank_dims.y - 1) rank.my_half_steps.y += half_steps_y % rank_dims.y;
    if(rank.coords_rank_space.z == rank_dims.z - 1) rank.my_half_steps.z += half_steps_z % rank_dims.z;

    if(rank.coords_rank_space.x > 0){
        rank.coords_sim_space.x -= 1;
        rank.my_half_steps.x += 1;
        rank.prevX = true;
    }
    if(rank.coords_rank_space.x < rank_dims.x - 1){
        rank.my_half_steps.x += 1;
        rank.nextX = true;
    }

    if(rank.coords_rank_space.y > 0){
        rank.coords_sim_space.y -= 1;
        rank.my_half_steps.y += 1;
        rank.prevY = true;
    }
    if(rank.coords_rank_space.y < rank_dims.y - 1){
        rank.my_half_steps.y += 1;
        rank.nextY = true;
    }

    if(rank.coords_rank_space.z > 0){
        rank.coords_sim_space.z -= 1;
        rank.my_half_steps.z += 1;
        rank.prevZ = true;
    }
    if(rank.coords_rank_space.z < rank_dims.z - 1){
        rank.my_half_steps.z += 1;
        rank.nextZ = true;
    }

    //first compute the number of local cells
    //TODO: Figure out how to do this more intelligently
    bool source_evenX = (rank.coords_sim_space.x % 2 == 0);
    bool source_evenY = (rank.coords_sim_space.y % 2 == 0);
    bool source_evenZ = (rank.coords_sim_space.z % 2 == 0);
    bool evenX = source_evenX;
    bool evenY = source_evenY;
    bool evenZ = source_evenZ;
    rank.num_local_cells = 0;
    for(int i = 0; i < rank.my_half_steps.x; i++){
        for(int j = 0; j < rank.my_half_steps.y; j++){
            for(int k = 0; k < rank.my_half_steps.z; k++){
                if(isPoint(evenX, evenY, evenZ)){
                    rank.num_local_cells++;
                }
                evenZ = !evenZ;
            }
            evenY = !evenY;
            evenZ = source_evenZ;
        }
        evenX = !evenX;
        evenY = source_evenY;
        evenZ = source_evenZ;
    }
    evenX = source_evenX;
    evenY = source_evenY;
    evenZ = source_evenZ;

    //initialize our local process node
    ProcessNode<FCCDiffuser> proc_node;
    proc_node.m_data = new DataNode<FCCDiffuser>[rank.num_local_cells];

    //These maps let us construct all the pointers we need for communication
    std::unordered_map<int, std::vector<int>> idx_map; // process idx -> vector of global indeces

    int idx = 0;
    for(int i = 0; i < rank.my_half_steps.x; i++){
        for(int j = 0; j < rank.my_half_steps.y; j++){
            for(int k = 0; k < rank.my_half_steps.z; k++){
                bool ghost = false;
                if(i == 0 && rank.prevX) ghost = true;
                if(j == 0 && rank.prevY) ghost = true;
                if(k == 0 && rank.prevZ) ghost = true;
                if(i == rank.my_half_steps.x-1 && rank.nextX) ghost = true;
                if(j == rank.my_half_steps.y-1 && rank.nextY) ghost = true;
                if(k == rank.my_half_steps.z-1 && rank.nextZ) ghost = true;
                if(isPoint(evenX, evenY, evenZ)){
                    proc_node.m_data[idx].m_ghost = ghost;
                    FCCDiffuser data_struct;
                    data_struct.x = rank.coords_sim_space.x + i;
                    data_struct.y = rank.coords_sim_space.y + j;
                    data_struct.z = rank.coords_sim_space.z + k;
                    data_struct.i = to1D({data_struct.x, data_struct.y, data_struct.z},
                                         {half_steps_x, half_steps_y, half_steps_z})/2;
                    data_struct.amount = 0.0f;
                    proc_node.m_data[idx].m_data = data_struct;
                    
                    //count up number of neighbors
                    proc_node.m_data[idx].m_num_neighbors = 0;
                    for(int dx = -1; dx <= 1; dx++){
                        for(int dy = -1; dy <= 1; dy++){
                            for(int dz = -1; dz <= 1; dz++){
                                int nz = 0;
                                if(dx == 0) nz++;
                                if(dy == 0) nz++;
                                if(dz == 0) nz++;
                                if(nz == 1){
                                    Point3D new_point = {proc_node.m_data[idx].m_data.x + dx,
                                                         proc_node.m_data[idx].m_data.y + dy,
                                                         proc_node.m_data[idx].m_data.z + dz};
                                    Point3D local = {new_point.x - rank.coords_sim_space.x,
                                                     new_point.y - rank.coords_sim_space.y,
                                                     new_point.z - rank.coords_sim_space.z};
                                    if(inBounds(local, rank.my_half_steps)){
                                        proc_node.m_data[idx].m_num_neighbors++;
                                    }
                                }
                            }
                        }
                    }
                    //set neighbors
                    proc_node.m_data[idx].m_neighbors = new DataNode<FCCDiffuser>*[proc_node.m_data[idx].m_num_neighbors];
                    //TODO: Properly free the m_neighbors list!!!
                    int temp = 0;
                    for(int dx = -1; dx <= 1; dx++){
                        for(int dy = -1; dy <= 1; dy++){
                            for(int dz = -1; dz <= 1; dz++){
                                int nz = 0;
                                if(dx == 0) nz++;
                                if(dy == 0) nz++;
                                if(dz == 0) nz++;
                                if(nz == 1){
                                    Point3D new_point = {proc_node.m_data[idx].m_data.x + dx,
                                                         proc_node.m_data[idx].m_data.y + dy,
                                                         proc_node.m_data[idx].m_data.z + dz};
                                    Point3D local = {new_point.x - rank.coords_sim_space.x,
                                                     new_point.y - rank.coords_sim_space.y,
                                                     new_point.z - rank.coords_sim_space.z};
                                    if(inBounds(local, rank.my_half_steps)){
                                        int nbr_idx = to1D(local, rank.my_half_steps)/2;
                                        proc_node.m_data[idx].m_neighbors[temp] = &proc_node.m_data[nbr_idx];
                                        temp++;
                                    }
                                }
                            }
                        }
                    }
                    if(ghost){
                        Point3D rank_pos = {proc_node.m_data[idx].m_data.x / half_steps_per_rank.x,
                                            proc_node.m_data[idx].m_data.y / half_steps_per_rank.y,
                                            proc_node.m_data[idx].m_data.z / half_steps_per_rank.z};
                        int pid = to1D(rank_pos, rank_dims);
                        int A = to1D({proc_node.m_data[idx].m_data.x,
                                      proc_node.m_data[idx].m_data.y,
                                      proc_node.m_data[idx].m_data.z},
                                      {half_steps_x, half_steps_y, half_steps_z});
                        if(idx_map.find(pid) == idx_map.end()){
                            idx_map.insert(std::make_pair(pid, std::vector<int>()));
                        } else {
                            idx_map.at(pid).push_back(A);
                        }
                    }
                    idx++;
                }
                evenZ = !evenZ;
            }
            evenY = !evenY;
            evenZ = source_evenZ;
        }
        evenX = !evenX;
        evenY = source_evenY;
        evenZ = source_evenZ;
    }
    evenX = source_evenX;
    evenY = source_evenY;
    evenZ = source_evenZ;

    //this is how to set the amount in a cell with global coordinates
    Point3D setPoint = {6, 6 ,6};
    Point3D localSetPoint = {setPoint.x - rank.coords_sim_space.x,
                             setPoint.y - rank.coords_sim_space.y,
                             setPoint.x - rank.coords_sim_space.z};
    if(inBounds(localSetPoint, rank.my_half_steps)){
        int idx = to1D(localSetPoint, rank.my_half_steps)/2;
        proc_node.m_data[idx].m_data.amount = 10000.0f;
    }

    //We need to tell each neighboring process how to pack data

    //write the topology to file for inspection
    //write output to file
    std::stringstream outputFileStream;
    outputFileStream << "./output/" << upcxx::rank_me() << "_log.csv";
    std::ofstream outputFile;
    outputFile.open(outputFileStream.str());
    outputFile << "x,y,z,i,ghost,nbrs" << std::endl;
    for(int i = 0; i < rank.num_local_cells; i++){
        FCCDiffuser point = proc_node.m_data[i].m_data;
        DataNode<FCCDiffuser> data = proc_node.m_data[i];
        outputFile << point.x << ",";
        outputFile << point.y << ",";
        outputFile << point.z << ",";
        outputFile << point.i << ",";
        outputFile << data.m_ghost << ",";

        for(int j = 0; j < data.m_num_neighbors; j++){
            outputFile << data.m_neighbors[j]->m_data.i << "|";
        }
        outputFile << std::endl;

    }

    // memory clean up
    delete[] proc_node.m_data;

    upcxx::barrier();
    upcxx::finalize();
    return 0;
}