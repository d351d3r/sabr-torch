#include <torch/torch.h>
#include <iostream>
#include <list>
#include <vector>

#include "LevenbergMarquad.hpp"
#include "sigma_SABR.hpp"


template <class Data>
auto data_points(Data data) {
  auto middle{[data](auto func){
    return [data, func](auto... args) { return func(data, args...); };
  }};

  return middle;
}

static torch::Tensor f(torch::Tensor t, torch::Tensor p) {
    return p[0] * exp(-t / p[1]) + p[2] * t * exp(-t / p[3]);
}

static auto K = torch::linspace(0.04, 0.11, 25);
        float S = 0.06,
        T = 0.5,
        alpha = 0.037561,
        beta = 0.5,
        rho = 0.100044,
        nu = 0.573296;

//sigma_SABR
//LevenbergMarquad lm();
int main() {
    torch::Tensor true_p = torch::randn({20, 10, 1, 50}); // True parameteres
    torch::Tensor x_true = torch::linspace(0, 100, 25); // span of free parameter
    torch::Tensor y_true = f(x_true, true_p); // fitted function observed values

    torch::Tensor init_p = true_p + pow((torch::randn(4) * 2), 2) + 4;
/// // initial guess for parametes = true + noise
    auto modified_f = data_points(x_true)(f);  // wrapped function (dependent on only parameters)
    std::cout << "Initial guess for optimizer " << init_p << std::endl;

// params = { 'x' : x_true,
//            'init_p': init_p,
//            'y' : y_true,
//            'func' : modified_f,  # model to be fitted
//            'sigma' : 4.0,    # std of datapoint
//            'lambda_lm' : 10, # starting lambda
//            'eps1' : 1e-3,    
//            'eps2' : 1e-3,
//            'eps3' : 1e-3,
//            'eps4' : 1e-3,
//            'lm_up' : 11,     # up coefficient
//            'lm_down' : 9     # down coefficient
//            }

//     LevenbergMarquad lm(params)     

    LevenbergMarquad lm(init_p,x_true, y_true, modified_f);
    for (int i = 0; i <= 1000; ++i)
        lm.step();

    std::list<float> p_hat;
    std::list<float> p;
    std::list<float> lmbd;

    for (int i = 0; i < 10; i++) {
        p.push_back(float(torch::linalg::norm(lm.step(),2)));
        p_hat.push_back(float(torch::linalg::norm(lm.p - true_p, auto opt_ord = 2)));
        lmbd.push_back(lm.lambda_lm);
    }

//TODO: Sigma
    torch::Tensor K = torch::linspace(0.04, 0.11, 25);
    float S = 0.06;
    float T = 0.5;
    float alpha = 0.037561;
    float beta = 0.5;
    float rho = 0.100044;
    float nu = 0.573296;

    init_p = torch::Tensor{[0.1] * 3};
    true_p = torch::Tensor([alpha, nu, rho]);
    torch::Tensor( y_mkt = sigma_SABR(K, true_p);

    auto modified_sabr = data_points(K),(sigma_SABR);


    LevenbergMarquad lm_sigma(x_true, init_p, y_true, modified_f);
    for (int i = 0; i <= 1000; ++i)
        lm.step();
}