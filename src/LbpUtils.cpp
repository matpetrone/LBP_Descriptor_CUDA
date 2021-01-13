////
//// Created by francesca on 12/01/21.
////
//
//#include "LbpUtils.h"
//#include <vector>
//#include <fstream>
//#include <iostream>
//#include <sstream>
//
//// Read image from CSV file
//std::vector<std::vector<int>> LbpUtils::getImageFromCsv(std::string& csv_file_name){
////    Read image values from CSV and store into vectors
////    std::string  file_name = "/home/francesca/CLionProjects/examplecv2/data.csv";
//    std::vector<std::vector<int>> vec;
//    std::ifstream data(csv_file_name);
//    std::string line;
//    while (std::getline(data, line)){
//        std::stringstream lineStream(line);
//        std::string cell;
//        std::vector<int> point;
//        while (getline(lineStream, cell, ','))
//            point.push_back(stod(cell));
//        vec.push_back(point);
//    }
//    std::cout << vec.size() << std::endl;
//    std::cout << vec[0].size() << std::endl;
//
//    return vec;
//}
//
