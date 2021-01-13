//
// Created by francesca on 12/01/21.
//

#ifndef LBP_DESCRIPTOR_CUDA_LBPUTILS_H
#define LBP_DESCRIPTOR_CUDA_LBPUTILS_H

#include <vector>
#include <string>

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
//#include <cstdarg>

char path[] = __FILE__;
std::string pp = std::string(path);
#define root_path pp.substr(0, (pp.substr(0, pp.find_last_of("/")).find_last_of("/")))


// Read image from CSV file
std::vector<std::vector<int>> getImageFromCsv(std::string& csv_file_name){
//    Read image values from CSV and store into vectors
//    std::string  file_name = "/home/francesca/CLionProjects/examplecv2/data.csv";
    std::cout << "read image from CSV" <<std::endl;
    csv_file_name = root_path + "/" + csv_file_name;
    std::cout << "Image path is: " << csv_file_name << '\n';
    std::vector<std::vector<int>> vec;
    std::ifstream data(csv_file_name);
    std::string line;
    while (std::getline(data, line)){
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<int> point;
        while (getline(lineStream, cell, ','))
            point.push_back(stod(cell));
        vec.push_back(point);
    }
    std::cout << "image height: " << vec.size() << std::endl;
    std::cout << "image width: " << vec[0].size() << std::endl;

    return vec;
};


#endif //LBP_DESCRIPTOR_CUDA_LBPUTILS_H
