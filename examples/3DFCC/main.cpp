
#include "unicubemaker.hpp"

#include <vector>
#include <upcxx/upcxx.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>

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

int localCoordsToGlobalIDX(Point3D local_coords,
                        Point3D local_dims,
                        Point3D sim_dims,
                        Rank rank){
    Point3D global_coords;

    int rank_offset_x = rank.coords_sim_space.x;
    int rank_offset_y = rank.coords_sim_space.y;
    int rank_offset_z = rank.coords_sim_space.z;

    global_coords.x = local_coords.x + rank_offset_x;
    global_coords.y = local_coords.y + rank_offset_y;
    global_coords.z = local_coords.z + rank_offset_z;

    return to1D(global_coords, sim_dims)/2;
}

int globalCoordstoRank(Point3D global_coords, Point3D half_steps_per_rank, Point3D rank_dims, Point3D sim_dims){
    Point3D proc_coords = {global_coords.x / half_steps_per_rank.x,
                            global_coords.y / half_steps_per_rank.y,
                            global_coords.z / half_steps_per_rank.z};
    return to1D(proc_coords, rank_dims);   
}


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
    Point3D sim_dims = {half_steps_x, half_steps_y, half_steps_z};
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
    proc_node.m_data_nodes = new DataNode<FCCDiffuser>[rank.num_local_cells];

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
                if(!isPoint(evenX, evenY, evenZ)){
                    idx++;
                    evenZ = !evenZ;
                    continue;
                }
                proc_node.m_data_nodes[idx].m_ghost = ghost;
                FCCDiffuser data_struct;
                data_struct.x = rank.coords_sim_space.x + i;
                data_struct.y = rank.coords_sim_space.y + j;
                data_struct.z = rank.coords_sim_space.z + k;
                data_struct.i = to1D({data_struct.x, data_struct.y, data_struct.z},
                                     {half_steps_x, half_steps_y, half_steps_z})/2;
                data_struct.amount = 0.0f;
                proc_node.m_data_nodes[idx].m_data = data_struct;
                
                //count up number of neighbors
                proc_node.m_data_nodes[idx].m_num_neighbors = 0;
                for(int dx = -1; dx <= 1; dx++){
                    for(int dy = -1; dy <= 1; dy++){
                        for(int dz = -1; dz <= 1; dz++){
                            int nz = 0;
                            if(dx == 0) nz++;
                            if(dy == 0) nz++;
                            if(dz == 0) nz++;
                            if(nz == 1){
                                Point3D new_point = {proc_node.m_data_nodes[idx].m_data.x + dx,
                                                     proc_node.m_data_nodes[idx].m_data.y + dy,
                                                     proc_node.m_data_nodes[idx].m_data.z + dz};
                                Point3D local = {new_point.x - rank.coords_sim_space.x,
                                                 new_point.y - rank.coords_sim_space.y,
                                                 new_point.z - rank.coords_sim_space.z};
                                if(inBounds(local, rank.my_half_steps)){
                                    proc_node.m_data_nodes[idx].m_num_neighbors++;
                                }
                            }
                        }
                    }
                }
                //set neighbors
                //TODO: Properly free the m_neighbors list!!!
                proc_node.m_data_nodes[idx].m_neighbors = new DataNode<FCCDiffuser>*[proc_node.m_data_nodes[idx].m_num_neighbors];
                int temp = 0;
                std::vector<int> proc_neighbors;
                for(int dx = -1; dx <= 1; dx++){
                    for(int dy = -1; dy <= 1; dy++){
                        for(int dz = -1; dz <= 1; dz++){
                            int nz = 0;
                            if(dx == 0) nz++;
                            if(dy == 0) nz++;
                            if(dz == 0) nz++;
                            if(nz == 1){
                                Point3D new_point = {proc_node.m_data_nodes[idx].m_data.x + dx,
                                                     proc_node.m_data_nodes[idx].m_data.y + dy,
                                                     proc_node.m_data_nodes[idx].m_data.z + dz};
                                Point3D local = {new_point.x - rank.coords_sim_space.x,
                                                 new_point.y - rank.coords_sim_space.y,
                                                 new_point.z - rank.coords_sim_space.z};
                                if(inBounds(local, rank.my_half_steps)){
                                    int nbr_idx = to1D(local, rank.my_half_steps)/2;
                                    proc_node.m_data_nodes[idx].m_neighbors[temp] = &proc_node.m_data_nodes[nbr_idx];
                                    temp++;
                                    bool nb_ghost = false;
                                    if(local.x == 0 && rank.prevX) nb_ghost = true;
                                    if(local.y == 0 && rank.prevY) nb_ghost = true;
                                    if(local.z == 0 && rank.prevZ) nb_ghost = true;
                                    if(local.x == rank.my_half_steps.x-1 && rank.nextX) nb_ghost = true;
                                    if(local.y == rank.my_half_steps.y-1 && rank.nextY) nb_ghost = true;
                                    if(local.z == rank.my_half_steps.z-1 && rank.nextZ) nb_ghost = true;
                                    if(nb_ghost){
                                        int global_idx = localCoordsToGlobalIDX(local,
                                                                            rank.my_half_steps,
                                                                            sim_dims,
                                                                            rank);
                                        Point3D global_coords = {rank.coords_sim_space.x + local.x,
                                                                rank.coords_sim_space.y + local.y,
                                                                rank.coords_sim_space.z + local.z};
                                        int owner_rank = globalCoordstoRank(global_coords,
                                                                        half_steps_per_rank,
                                                                        rank_dims, sim_dims);
                                        proc_neighbors.push_back(owner_rank);
                                    }
                                }
                            }
                        }
                    }
                }
                for(unsigned int zz = 0; zz < proc_neighbors.size(); zz++){
                    if(upcxx::rank_me() == 0) std::cout << proc_neighbors[zz] << std::endl;
                    int proc_id = proc_neighbors[zz];
                    if(proc_node.m_pack_map.count(proc_id) == 0){
                        proc_node.m_pack_map.insert( std::make_pair( proc_id, std::vector<int>() ) );
                    }
                    proc_node.m_pack_map.at(proc_id).push_back(idx);
                }

                if(proc_node.m_data_nodes[idx].m_ghost){
                    int global_idx = localCoordsToGlobalIDX({i, j, k},
                                                        rank.my_half_steps,
                                                        sim_dims,
                                                        rank);
                    Point3D global_coords = {rank.coords_sim_space.x + i,
                                            rank.coords_sim_space.y + j,
                                            rank.coords_sim_space.z + k};
                    int owner_rank = globalCoordstoRank(global_coords,
                                                    half_steps_per_rank,
                                                    rank_dims, sim_dims);
                    if(proc_node.m_unpack_map.count(owner_rank) == 0){
                        proc_node.m_unpack_map.insert( std::make_pair( owner_rank, std::vector<int>() ) );
                    }
                    proc_node.m_unpack_map.at(owner_rank).push_back(idx);
                }
                idx++;
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

    // //get the sizes of packed data
    // for(auto pair : proc_node.m_pack_map){
    //     int proc_id = pair.first;
    //     proc_node.m_packed_data_sizes[proc_id] = pair.second.size();
    // }

    //this is how to set the amount in a cell with global coordinates
    Point3D setPoint = {6, 6 ,6};
    Point3D localSetPoint = {setPoint.x - rank.coords_sim_space.x,
                             setPoint.y - rank.coords_sim_space.y,
                             setPoint.x - rank.coords_sim_space.z};
    if(inBounds(localSetPoint, rank.my_half_steps)){
        int idx = to1D(localSetPoint, rank.my_half_steps)/2;
        proc_node.m_data_nodes[idx].m_data.amount = 10000.0f;
    }

    //write the topology to file for inspection
    //write output to file
    std::stringstream outputFileStream;
    outputFileStream << "./output/" << upcxx::rank_me() << "_log.csv";
    std::ofstream outputFile;
    outputFile.open(outputFileStream.str());
    outputFile << "x,y,z,i,ghost,nbrs" << std::endl;
    for(int i = 0; i < rank.num_local_cells; i++){
        FCCDiffuser fccData = proc_node.m_data_nodes[i].m_data;
        DataNode<FCCDiffuser> dnode = proc_node.m_data_nodes[i];
        int global_idx = localCoordsToGlobalIDX({fccData.x, fccData.y, fccData.z},
                                            rank.my_half_steps,
                                            sim_dims,
                                            rank);
        Point3D point = {rank.coords_sim_space.x + fccData.x,
                        rank.coords_sim_space.y + fccData.y,
                        rank.coords_sim_space.z + fccData.z};
        outputFile << point.x << ",";
        outputFile << point.y << ",";
        outputFile << point.z << ",";
        outputFile << global_idx << ",";
        outputFile << dnode.m_ghost << ",";

        for(int j = 0; j < dnode.m_num_neighbors - 1; j++){
            int nbr_global_idx = localCoordsToGlobalIDX({dnode.m_neighbors[j]->m_data.x,
                                                        dnode.m_neighbors[j]->m_data.y,
                                                        dnode.m_neighbors[j]->m_data.z}, rank.my_half_steps, sim_dims, rank);
            outputFile << nbr_global_idx << "|";
        }
        if(dnode.m_num_neighbors > 0){
            int nbr_global_idx = localCoordsToGlobalIDX({dnode.m_neighbors[dnode.m_num_neighbors - 1]->m_data.x,
                                            dnode.m_neighbors[dnode.m_num_neighbors - 1]->m_data.y,
                                            dnode.m_neighbors[dnode.m_num_neighbors - 1]->m_data.z}, rank.my_half_steps, sim_dims, rank);
            outputFile << nbr_global_idx;
        }
        outputFile << std::endl;

    }

    // memory clean up
    delete[] proc_node.m_data_nodes;

    upcxx::barrier();
    upcxx::finalize();
    return 0;
}
