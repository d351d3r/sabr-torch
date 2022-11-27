#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <list>
#include <vector>

#include "LevenbergMarquad.hpp"
#include "sigma_SABR.hpp"

// template <class Data>
// auto data_points(Data data) {
//   auto middle{[data](auto func) {
//     return [data, func](auto... args) { return func(data, args...); };
//   }};

//   return middle;
// }

static torch::Tensor f(torch::Tensor t, torch::Tensor p) {
  return p[0] * exp(-t / p[1]) + p[2] * t * exp(-t / p[3]);
}

struct LM_PARAMS {
  torch::Tensor x;
  torch::Tensor init_p;
  torch::Tensor y;
  torch::Tensor func;
  float sigma = 4.0; // std of datapoint
  int lambda_lm = 10; // starting lambda
  float eps1 = 1e-3;
  float eps2 = 1e-3;
  float eps3 = 1e-3;
  float eps4 = 1e-3;
  int lm_up = 11; // up coefficient
  int lm_down = 9; // down coefficient
};

int main() {
  std::fstream strm;
  strm.open("input.txt", std::ios_base::out);

  if (strm.is_open()) {
    std::string str;
    strm << str;
    strm.close();
  } else
    std::cerr << "File could not be opened!" << std::endl;

  // TODO Input from file (matrix)
  // striike,val,sigm
  
  torch::Tensor true_p = torch::randn({20, 10, 1, 50}); // True parameteres
  torch::Tensor x_true = torch::linspace(0, 100, 25); // span of free parameter
  torch::Tensor y_true = f(x_true, true_p); // fitted function observed values

  torch::Tensor init_p = true_p + pow((torch::randn(4) * 2), 2) + 4;
  /// // initial guess for parametes = true + noise
  auto modified_f =
      data_points(x_true)(f); // wrapped function (dependent on only parameters)
  std::cout << "Initial guess for optimizer " << init_p << std::endl;

  //     LevenbergMarquad lm(params)
  torch::Tensor f;
  LM_PARAMS lm_params{};
  LevenbergMarquad lm(
      init_p,
      x_true,
      y_true,
      modified_f,
      lm_params.sigma,
      lm_params.lambda_lm,
      lm_params.eps1,
      lm_params.eps2,
      lm_params.eps3,
      lm_params.eps4);

  for (int i = 0; i <= 1000; ++i)
    lm.step();

  std::list<float> p_hat;
  std::list<float> p;
  std::list<float> lmbd;

  for (int i = 0; i < 10; i++) {
    p.push_back(float(torch::linalg::norm(lm.step(), 2)));
    p_hat.push_back(
        float(torch::linalg::norm(lm.p - true_p, auto opt_ord = 2)));
    lmbd.push_back(lm.lambda_lm);
  }

  // TODO: Sigma
  torch::Tensor K = torch::linspace(0.04, 0.11, 25);
  float S = 0.06;
  float T = 0.5;
  float alpha = 0.037561;
  float beta = 0.5;
  float rho = 0.100044;
  float nu = 0.573296;

  init_p = torch::randn({0.1 * 3});
  true_p = torch::randn([ alpha, nu, rho ]);
    torch::Tensor(y_mkt = sigma_SABR(K, true_p);


// TODO
    auto modified_sabr = data_points(K),(sigma_SABR);
// TODO

    struct SIGMA_PARAMS : LM_PARAMS {torch::Tensor x = K,modified_sabr([ alpha, nu, rho]), func = modified_sabr, 
    sigma = 1.0,lambda_lm = 0.1, lm_up = 10, lm_down = 9};
   
    // TODO
    LevenbergMarquad lm_sigma(x_true, init_p, y_true, modified_f);
    for (int i = 0; i <= 1000; ++i)
        lm.step();
}
