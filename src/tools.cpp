#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  
  
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  if (estimations.size() == 0){
      std::cout << "Error: no estimations vector!" << endl;
      return rmse;
  }
  
  if (estimations.size() != ground_truth.size()){
      std::cout << "Error: Vector size mismatch!" << endl;
      return rmse;
  }   

  for(int i=0; i < estimations.size(); ++i){
      VectorXd diff = (estimations[i] - ground_truth[i]).array().pow(2);
      rmse += diff;
  }

  rmse /= estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;  
}
