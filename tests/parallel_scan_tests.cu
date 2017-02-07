#include <gmock/gmock.h>
#include <iostream>
#include <fstream>
#include <mg_gpgpu_core/parallel_scan.h>
#include <vector>
using namespace testing;

//commented for the time being, need to find a way to make a custom target for it
//void inclusive_scan(std::vector<unsigned int>& data)
//{
//
//    for (int i =1; i<data.size(); ++i)
//    {
//        data[i] += data[i-1];    
//    }
//
//}
//
//TEST(cuda_parallel_scan, inital_test)
//{
//    int count =0;
//    while(true)
//    {
//        std::vector<unsigned int > data;
//        std::vector<unsigned int > original;
//        unsigned int size = rand() %(100000) ;
//        //std::cout<<size<<std::endl;
//        data.resize(size);
//        original.resize(size);
//        for (int i =0 ; i <size; ++i)
//        {
//            data[i] = rand() % 2 + 1;
//            original[i] = data[i];
//            //if (i <=2 )
//            //{
//            //    std::cout<<data[i]<<std::endl; 
//            //}
//        }
//
//        //std::cout<<"calling_cuda"<<std::endl;
//        auto ptr = data.data();
//        auto cudares = parallel_scan(ptr, size);
//        inclusive_scan(data);
//        for(int i =1; i < size; ++i)
//        {
//            
//           if( data[i] != cudares[i])
//           {
//               
//                std::cout<<"iter "<<count<<" ERROR AT INDEX: "<<i<<", "<<data[i]<<" "<<cudares[i]<<std::endl; 
//                std::string output = "original = [";
//
//                for ( auto d: original)
//                {
//                    output += std::to_string(d) + ", "; 
//                }
//
//                output += "] \n cpu = [";
//                for ( auto d: data)
//                {
//                    output += std::to_string(d) + ", "; 
//                }
//                output += "] \n gpu = [";
//                for (int i =0 ; i<size; ++i)
//                {
//                    output += std::to_string(cudares[i] ) +", ";
//                
//                }
//                std::string path = "output.txt";
//                std::ofstream out(path);
//                out << output;
//                out.close();
//                ASSERT_TRUE(0);
//           } 
//        
//        }
//        ++count;
//    }
//}

