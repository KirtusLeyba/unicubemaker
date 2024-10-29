
#include "unicubemaker.hpp"

#include <vector>
#include <upcxx/upcxx.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <set>

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

//within boundary e.g. x in [min, max)
bool inBounds(Point3D p, Point3D min, Point3D max){
    if(p.x >= max.x || p.x < min.x) return false;
    if(p.y >= max.y || p.y < min.y) return false;
    if(p.z >= max.z || p.z < min.z) return false;
    return true;
}

int getRankOwner(Point3D p, Point3D half_steps_per_rank, Point3D rank_dims){
    int rx = p.x / half_steps_per_rank.x;
    if(rx >= rank_dims.x) rx = rank_dims.x - 1;
    int ry = p.y / half_steps_per_rank.y;
    if(ry >= rank_dims.y) ry = rank_dims.y - 1;
    int rz = p.z / half_steps_per_rank.z;
    if(rz >= rank_dims.z) rz = rank_dims.z - 1;
    return to1D({rx, ry, rz}, rank_dims);
}

struct FCCDiffuser {
    int x;
    int y;
    int z;
    int i;
    float amount;
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
    Point3D sim_dims = {8, 8, 8};
    int num_ranks = upcxx::rank_n();
    Point3D rank_dims = {2, 1, 1};

    bool prevX = false;
    bool nextX = false;
    bool prevY = false;
    bool nextY = false;
    bool prevZ = false;
    bool nextZ = false;
    Point3D half_steps_per_rank = { sim_dims.x / rank_dims.x,
                                    sim_dims.y / rank_dims.y,
                                    sim_dims.z / rank_dims.z };
    Point3D my_half_steps = { half_steps_per_rank.x,
                            half_steps_per_rank.y,
                            half_steps_per_rank.z};
    Point3D coords_rank_space = to3D(upcxx::rank_me(), rank_dims);
    Point3D coords_sim_space_start = {coords_rank_space.x*half_steps_per_rank.x,
                                      coords_rank_space.y*half_steps_per_rank.y,
                                      coords_rank_space.z*half_steps_per_rank.z};
    if(coords_rank_space.x == rank_dims.x - 1) my_half_steps.x += sim_dims.x % rank_dims.x;
    if(coords_rank_space.y == rank_dims.y - 1) my_half_steps.y += sim_dims.y % rank_dims.y;
    if(coords_rank_space.z == rank_dims.z - 1) my_half_steps.z += sim_dims.z % rank_dims.z;
    Point3D coords_sim_space_end = {coords_sim_space_start.x + my_half_steps.x,
                                    coords_sim_space_start.y + my_half_steps.y,
                                    coords_sim_space_start.z + my_half_steps.z};


    if(coords_rank_space.x > 0){
        prevX = true;
        coords_sim_space_start.x -= 1;
    }
    if(coords_rank_space.x < rank_dims.x - 1){
        nextX = true;
        coords_sim_space_end.x += 1;
    }

    if(coords_rank_space.y > 0){
        prevY = true;
        coords_sim_space_start.y -= 1;
    }
    if(coords_rank_space.y < rank_dims.y - 1){
        nextY = true;
        coords_sim_space_end.y += 1;
    }

    if(coords_rank_space.z > 0){
        prevZ = true;
        coords_sim_space_start.z -= 1;
    }
    if(coords_rank_space.z < rank_dims.z - 1){
        nextZ = true;
        coords_sim_space_end.z += 1;
    }

    bool evenX = true;
    bool evenY = true;
    bool evenZ = true;
    int num_local_cells = 0;
    std::unordered_map<int, Point3D> idx_to_xyz;
    std::unordered_map< std::tuple<int, int, int>,
                        int,
                        boost::hash<std::tuple<int, int, int>> > xyz_to_idx;
    std::unordered_map<int, int> global_to_local;
    int idx = 0;
    int local_idx = 0;
    for(int i = 0; i < sim_dims.x; i++){
        for(int j = 0; j < sim_dims.y; j++){
            for(int k = 0; k < sim_dims.z; k++){
                if(isPoint(evenX, evenY, evenZ)){
                    if(inBounds({i, j, k}, coords_sim_space_start, coords_sim_space_end)){
                        num_local_cells++;
                        idx_to_xyz[idx] = {i, j, k};
                        xyz_to_idx[std::make_tuple(i, j, k)] = idx;
                        global_to_local[idx] = local_idx;
                        local_idx += 1;
                    }
                    idx += 1;
                }
                evenZ = !evenZ;
            }
            evenY = !evenY;
            evenZ = true;
        }
        evenX = !evenX;
        evenY = true;
        evenZ = true;
    }
    evenX = true;
    evenY = true;
    evenZ = true;

    //initialize our local process node
    ProcessNode<FCCDiffuser> proc_node;
    proc_node.m_data_nodes = new DataNode<FCCDiffuser>[num_local_cells];
    idx = 0;
    local_idx = 0;
    int ghosts_seen = 0;
    for(int i = 0; i < sim_dims.x; i++){
        for(int j = 0; j < sim_dims.y; j++){
            for(int k = 0; k < sim_dims.z; k++){
                bool ghost = false;
                if(i == coords_sim_space_start.x && prevX) ghost = true;
                if(j == coords_sim_space_start.y && prevY) ghost = true;
                if(k == coords_sim_space_start.z && prevZ) ghost = true;
                if(i == coords_sim_space_end.x-1 && nextX) ghost = true;
                if(j == coords_sim_space_end.y-1 && nextY) ghost = true;
                if(k == coords_sim_space_end.z-1 && nextZ) ghost = true;

                if(!inBounds({i, j, k}, coords_sim_space_start, coords_sim_space_end)){
                    if(isPoint(evenX, evenY, evenZ)) idx++;
                    evenZ = !evenZ;
                    continue;
                }
                if(!isPoint(evenX, evenY, evenZ)){
                    evenZ = !evenZ;
                    continue;
                }
                proc_node.m_data_nodes[local_idx].m_ghost = ghost;
                FCCDiffuser data_struct;
                data_struct.x = i;
                data_struct.y = j;
                data_struct.z = k;
                data_struct.i = xyz_to_idx.at(std::make_tuple(i, j, k));
                data_struct.amount = 0.0f;
                proc_node.m_data_nodes[local_idx].m_data = data_struct;

                //count up number of neighbors
                proc_node.m_data_nodes[local_idx].m_num_neighbors = 0;
                for(int dx = -1; dx <= 1; dx++){
                    for(int dy = -1; dy <= 1; dy++){
                        for(int dz = -1; dz <= 1; dz++){
                            int nz = 0;
                            if(dx == 0) nz++;
                            if(dy == 0) nz++;
                            if(dz == 0) nz++;
                            if(nz == 1){
                                Point3D new_point = {proc_node.m_data_nodes[local_idx].m_data.x + dx,
                                                     proc_node.m_data_nodes[local_idx].m_data.y + dy,
                                                     proc_node.m_data_nodes[local_idx].m_data.z + dz};
                                if(inBounds(new_point, coords_sim_space_start, coords_sim_space_end)){
                                    proc_node.m_data_nodes[local_idx].m_num_neighbors++;
                                }
                            }
                        }
                    }
                }
                //set neighbors
                //TODO: Properly free the m_neighbors list!!!
                proc_node.m_data_nodes[local_idx].m_neighbors = new DataNode<FCCDiffuser>*[proc_node.m_data_nodes[local_idx].m_num_neighbors];
                int temp = 0;
                std::set<int> proc_neighbors;
                for(int dx = -1; dx <= 1; dx++){
                    for(int dy = -1; dy <= 1; dy++){
                        for(int dz = -1; dz <= 1; dz++){
                            int nz = 0;
                            if(dx == 0) nz++;
                            if(dy == 0) nz++;
                            if(dz == 0) nz++;
                            if(nz == 1){
                                Point3D new_point = {proc_node.m_data_nodes[local_idx].m_data.x + dx,
                                                     proc_node.m_data_nodes[local_idx].m_data.y + dy,
                                                     proc_node.m_data_nodes[local_idx].m_data.z + dz};
                                if(inBounds(new_point, coords_sim_space_start, coords_sim_space_end)){
                                    int nbr_global_idx = xyz_to_idx.at(std::make_tuple(new_point.x,
                                                                                       new_point.y,
                                                                                       new_point.z));
                                    int nbr_idx = global_to_local.at(nbr_global_idx);
                                    proc_node.m_data_nodes[local_idx].m_neighbors[temp] = &proc_node.m_data_nodes[nbr_idx];
                                    temp++;
                                    bool nb_ghost = false;
                                    if(new_point.x == coords_sim_space_start.x && prevX) nb_ghost = true;
                                    if(new_point.y == coords_sim_space_start.y && prevY) nb_ghost = true;
                                    if(new_point.z == coords_sim_space_start.z && prevZ) nb_ghost = true;
                                    if(new_point.x == coords_sim_space_end.x-1 && nextX) nb_ghost = true;
                                    if(new_point.y == coords_sim_space_end.y-1 && nextY) nb_ghost = true;
                                    if(new_point.z == coords_sim_space_end.z-1 && nextZ) nb_ghost = true;
                                    if(nb_ghost && !ghost){
                                        int owner_rank = getRankOwner(new_point, half_steps_per_rank, rank_dims);
                                        proc_neighbors.insert(owner_rank);
                                    }
                                }
                            }
                        }
                    }
                }
                if(!ghost){
                    for(const int &proc_id : proc_neighbors){
                        if(proc_node.m_pack_map.count(proc_id) == 0){
                            proc_node.m_pack_map.insert( std::make_pair( proc_id, std::vector<int>() ) );
                        }
                        proc_node.m_pack_map.at(proc_id).push_back(local_idx);
                    }
                }

                if(proc_node.m_data_nodes[local_idx].m_ghost){
                    int owner_rank = getRankOwner(idx_to_xyz.at(idx), half_steps_per_rank, rank_dims);
                    if(proc_node.m_unpack_map.count(owner_rank) == 0){
                        proc_node.m_unpack_map.insert( std::make_pair( owner_rank, std::vector<int>() ) );
                    }
                    proc_node.m_unpack_map.at(owner_rank).push_back(local_idx);
                }
                local_idx++;
                idx++;
                evenZ = !evenZ;
            }
            evenY = !evenY;
            evenZ = true;
        }
        evenX = !evenX;
        evenY = true;
        evenZ = true;
    }
    evenX = true;
    evenY = true;
    evenZ = true;


    //this is how to set the amount in a cell with global coordinates
    Point3D setPoint = {6, 6 ,6};
    if(inBounds(setPoint, coords_sim_space_start, coords_sim_space_end)){
        int idx = global_to_local.at(xyz_to_idx.at(std::make_tuple(setPoint.x, setPoint.y, setPoint.z)));
        proc_node.m_data_nodes[idx].m_data.amount = 10000.0f;
    }

    //setup comms network
    for(auto it : proc_node.m_pack_map){
        int proc_id = it.first;
        std::vector<int> locations = it.second;
        proc_node.m_packed_data[proc_id] = upcxx::new_array<DataNode<FCCDiffuser>>(locations.size());
        proc_node.m_packed_data_sizes[proc_id] = locations.size();
    }

    upcxx::barrier();
    proc_node.bcastGPTRs();
    upcxx::barrier();
    for(int i = 0; i < upcxx::rank_n(); i++){
        if(upcxx::rank_me() == i){
            std::cout << "rank: " << upcxx::rank_me() << std::endl;
            std::cout << "ghosts = " << ghosts_seen << std::endl;
            std::cout << "printing m_neighbor_data" << std::endl;
            for(auto it : proc_node.m_neighbor_data){
                int size = proc_node.m_neighbor_data_sizes.at(it.first);
                std::cout << it.first << ": " << it.second << " " << size << std::endl;
            }
            std::cout << "printing m_unpack_map" << std::endl;
            for(auto it : proc_node.m_unpack_map){
                std::cout << it.first << ": " << it.second.size() << std::endl;
            }
            std::cout << "printing m_pack_map" << std::endl;
            for(auto it : proc_node.m_pack_map){
                std::cout << it.first << ": " << it.second.size() << std::endl;
            }
        }
        upcxx::barrier();
    }

    //run a diffusion simulation
    //write output to file
    std::stringstream outputFileStream;
    outputFileStream << "./output/" << upcxx::rank_me() << "_log.csv";
    std::ofstream outputFile;
    outputFile.open(outputFileStream.str());
    outputFile << "it,x,y,z,i,amount,ghost" << std::endl;
    int num_steps = 10000;
    float D = 0.01;
    for(int ts = 0; ts < num_steps; ts++){
        DataNode<FCCDiffuser>* new_data = new DataNode<FCCDiffuser>[num_local_cells];

        //diffusion
        for(int i = 0; i < num_local_cells; i++){
            DataNode<FCCDiffuser>* dn = &(proc_node.m_data_nodes[i]);
            if(dn->m_ghost) continue;
            new_data[i].m_data = dn->m_data;
            for(int j = 0; j < dn->m_num_neighbors; j++){
                new_data[i].m_data.amount += D*dn->m_neighbors[j]->m_data.amount;
            }
            new_data[i].m_data.amount -= D*(float)dn->m_num_neighbors*dn->m_data.amount;
        }
        //swap vectors
        for(int i = 0; i < num_local_cells; i++){
            proc_node.m_data_nodes[i].m_data = new_data[i].m_data;
            outputFile << ts << "," << new_data[i].m_data.x << "," <<
                                        new_data[i].m_data.y << "," <<
                                        new_data[i].m_data.z << "," <<
                                        new_data[i].m_data.i << "," <<
                                        new_data[i].m_data.amount << "," <<
                                        new_data[i].m_ghost << std::endl;
        }
        delete[] new_data;

        //communicate
        proc_node.packData();
        upcxx::barrier();
        proc_node.recvAndUnpack();

    }
    outputFile.close();

    // memory clean up
    delete[] proc_node.m_data_nodes;

    upcxx::barrier();
    upcxx::finalize();
    return 0;
}
